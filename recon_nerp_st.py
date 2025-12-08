import os
import argparse
import time
import torch
import numpy as np
import tifffile as tif
from dahuffman import HuffmanCodec
from torch.cuda.amp import autocast


from model_box import NeRPSTPro, NeRPSTProDecoder
from auxiliary import (
    ndarray2tif_min_max_clip,
    ndarray2tif_mean_std_clip,
    video_resize,
    CONVERT_UINT16_FLOAT64,
    ndarray2tif_mean_max_clip,
    create_overlap_patch_info_test,
    get_interp_coord,
)


def dequant_tensor(quant_t, dtype=torch.float32, device=None):
    """
    De-quantize a tensor stored as {'quant', 'min', 'scale'}
    """
    quant = quant_t["quant"].to(dtype=dtype, device=device)
    tmin = quant_t["min"].to(dtype=dtype, device=device)
    scale = quant_t["scale"].to(dtype=dtype, device=device)
    return tmin.expand_as(quant) + scale.expand_as(quant) * quant


def recover_from_huffman(enc_dict, dec_list):
    """(Unused here) Recover multiple tensors from a flat decoded list."""
    recovered_len = 0
    recovered_ten = torch.tensor(dec_list)
    embed_len = torch.prod(torch.tensor(enc_dict["embed"])).item()
    quant_embed = {
        "quant": recovered_ten[recovered_len:recovered_len + embed_len].reshape(
            enc_dict["embed"]
        )
    }
    recovered_len += embed_len
    enc_dict.pop("embed")
    dec_dict = {}
    for k, v in enc_dict.items():
        cur_shape = v
        cur_len = torch.prod(torch.tensor(cur_shape)).item()
        dec_dict[k] = recovered_ten[recovered_len:recovered_len + cur_len].reshape(v)
        recovered_len += cur_len
    return quant_embed, dec_dict


def decode_huffman(huff_path, storage_name):
    """(Unused here) Decode Huffman-compressed embeddings."""
    codec = HuffmanCodec.load(os.path.join(huff_path, f"codec_{storage_name}"))
    with open(os.path.join(huff_path, f"encode_{storage_name}.bin"), "rb") as f:
        huff_emb = f.read()
    dec = codec.decode(huff_emb)
    dec_tensor = torch.tensor(dec).to(torch.uint8)
    uncompressed_tensor = torch.zeros(size=(2, len(dec_tensor)), dtype=torch.uint8)
    uncompressed_tensor[0] = (dec_tensor << 4) & 0x0F
    uncompressed_tensor[1] = dec_tensor >> 4
    return uncompressed_tensor


def main():
    parser = argparse.ArgumentParser(description="Fast Necoda reconstruction with GPU accumulation + AMP + torch.compile.")
    parser.add_argument(
        "-d", "--ckpt_store_dir",
        type=str,
        required=True,
        help="path for raw ckpt and reconstruction storage.",
    )
    parser.add_argument(
        "-e", "--epoch",
        type=int,
        required=True,
        help="select corresponding ckpt.",
    )
    parser.add_argument(
        "--frames",
        type=int,
        default=32,
        help="video frames for output (unused, kept for compatibility)",
    )
    parser.add_argument(
        "--final_size",
        type=int,
        nargs="+",
        default=[],
        help="final reconstruction size, shape in [h, w] (unused, kept for compatibility)",
    )
    parser.add_argument(
        "--name",
        type=str,
        default="recon",
        help="reconstructed file name (without extension)",
    )
    parser.add_argument(
        "-s", "--standard_range",
        action="store_true",
        default=False,
        help="if using standard range, then the image will be rescaled by dividing 65535, "
             "else using min-max scale. (kept for compatibility)",
    )
    parser.add_argument(
        "--tif_max",
        type=float,
        default=CONVERT_UINT16_FLOAT64,
        help="max value of raw tiff (kept for compatibility)",
    )
    parser.add_argument(
        "--tif_min",
        type=float,
        default=0,
        help="min value of raw tiff (kept for compatibility)",
    )
    parser.add_argument(
        "-g", "--generalization_id_list",
        type=int,
        nargs="+",
        default=[],
        help="Add generalization embedding and decoding, e.g. -g 1 2 3",
    )
    parser.add_argument(
        "--do_resize",
        action="store_true",
        default=False,
        help="Specify whether do resize for the dataset",
    )
    parser.add_argument(
        "--x_resize",
        type=int,
        default=0,
        help="Specify resized x",
    )
    parser.add_argument(
        "--y_resize",
        type=int,
        default=0,
        help="Specify resized y",
    )
    parser.add_argument(
        "--patch_size",
        type=int,
        default=1,
        help="number of patches decoded in a batch (batch size over patches)",
    )

    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    print("ckpt_store_dir:", args.ckpt_store_dir)
    print("Current epoch:", args.epoch)
    print("Generalization IDs:", args.generalization_id_list or [1])
    print("Batch patch_size (batch over patches):", args.patch_size)
    torch.backends.cudnn.benchmark = True
    if device.type == "cuda":
        if hasattr(torch.backends, "cuda") and hasattr(torch.backends.cuda, "matmul"):
            torch.backends.cuda.matmul.allow_tf32 = True
        if hasattr(torch.backends.cudnn, "allow_tf32"):
            torch.backends.cudnn.allow_tf32 = True

    if device.type != "cuda":
        print("WARNING: running on CPU, AMP & torch.compile will be disabled and it will be very slow.")

    net_storage_path = os.path.join(args.ckpt_store_dir, "g1")
    quant_all_base = torch.load(
        os.path.join(net_storage_path, f"quant_all_{args.epoch}.pth"),
        map_location="cpu",
    )
    print("quant_all_base.keys():", quant_all_base.keys())
    args_dict = quant_all_base["m_args"]  # typically a Namespace-like object

    model_all = torch.load(
        os.path.join(args.ckpt_store_dir, f"model_quant_{args.epoch}.pth"),
        map_location="cpu",
    )

    model = NeRPSTPro(
        raw_size_x=args_dict.patch_x,
        raw_size_t=args_dict.patch_t,
        interp_size_x=args_dict.interp_size_x * 2 + args_dict.patch_x,
        interp_size_t=args_dict.interp_size_t * 2 + args_dict.patch_t,
        interp_chn=args_dict.interp_chn,
        pre_s_rate=args_dict.pre_s_rate,
        pre_t_rate=args_dict.pre_t_rate,
        s_embedding_dim=args_dict.s_emb_dim,
        t_embedding_dim=args_dict.t_emb_dim,
        s_s_rate_list=args_dict.s_s_rate_list,
        s_t_rate_list=args_dict.s_t_rate_list,
        t_s_rate_list=args_dict.t_s_rate_list,
        t_t_rate_list=args_dict.t_t_rate_list,
        chns_list=args_dict.chns_list,
    )
    img_decoder = NeRPSTProDecoder(model).to(device)
    if device.type == "cuda":
        dec_state = {
            (k if "pass_way." not in k else k.replace("pass_way.", "")):
                dequant_tensor(v, dtype=torch.float16, device=device)
            for k, v in model_all.items()
        }
        img_decoder.load_state_dict(dec_state)
        img_decoder.half()
    else:
        dec_state = {
            (k if "pass_way." not in k else k.replace("pass_way.", "")):
                dequant_tensor(v, dtype=torch.float32, device=device)
            for k, v in model_all.items()
        }
        img_decoder.load_state_dict(dec_state)

    img_decoder.eval()
    if device.type == "cuda" and hasattr(torch, "compile"):
        try:
            img_decoder = torch.compile(img_decoder, mode="reduce-overhead")
            print("Using torch.compile on decoder (mode=reduce-overhead).")
        except Exception as e:
            print("torch.compile failed, fallback to eager. Error:", repr(e))
    else:
        if not hasattr(torch, "compile"):
            print("torch.compile not found in this PyTorch version, using eager.")

    print()
    print("Model:")
    print(model)
    print()
    if hasattr(torch, "inference_mode"):
        infer_ctx = torch.inference_mode
    else:
        infer_ctx = torch.no_grad
    gen_id_list = args.generalization_id_list or [1]

    for item in gen_id_list:
        print("=" * 80)
        print("Current generalization id: g{}".format(item))
        print("=" * 80)
        total_start_time = time.perf_counter()
        cur_ckpt_store_dir = os.path.join(args.ckpt_store_dir, f"g{item}")
        quant_all = torch.load(
            os.path.join(cur_ckpt_store_dir, f"quant_all_{args.epoch}.pth"),
            map_location="cpu",
        )
        if device.type == "cuda":
            vid_embed_s = dequant_tensor(quant_all["quant_embed_s"], dtype=torch.float16, device=device)
            vid_embed_t = dequant_tensor(quant_all["quant_embed_t"], dtype=torch.float16, device=device)
        else:
            vid_embed_s = dequant_tensor(quant_all["quant_embed_s"], dtype=torch.float32, device=device)
            vid_embed_t = dequant_tensor(quant_all["quant_embed_t"], dtype=torch.float32, device=device)
        args_dict_m = quant_all["m_args"]
        # Video shape
        T = args_dict_m.t
        X = args_dict_m.x
        Y = args_dict_m.y
        print(f"Video shape: T={T}, X={X}, Y={Y}")
        print("vid_embed_s.shape:", tuple(vid_embed_s.shape))
        print("vid_embed_t.shape:", tuple(vid_embed_t.shape))
        res = torch.zeros((T, X, Y), dtype=torch.float32, device=device)
        weight = torch.zeros_like(res)
        coord_list = create_overlap_patch_info_test(
            args_dict_m.patch_x,
            args_dict_m.x,
            args_dict_m.patch_t,
            args_dict_m.t,
        )
        num_patches = vid_embed_s.shape[0]
        assert len(coord_list) == num_patches, f"coord_list length {len(coord_list)} != num_patches {num_patches}"
        print("Number of patches:", num_patches)

        patch_size = max(1, args.patch_size)
        overlap_kernel = None

        t_ranges = []
        x_ranges = []
        y_ranges = []
        for (t_idx, x_idx, y_idx) in coord_list:
            st, et = get_interp_coord(
                t_idx,
                args_dict_m.t,
                args_dict_m.patch_t,
                args_dict_m.interp_size_t,
            )
            sx, ex = get_interp_coord(
                x_idx,
                args_dict_m.x,
                args_dict_m.patch_x,
                args_dict_m.interp_size_x,
            )
            sy, ey = get_interp_coord(
                y_idx,
                args_dict_m.y,
                args_dict_m.patch_y,
                args_dict_m.interp_size_x,
            )
            t_ranges.append((st, et))
            x_ranges.append((sx, ex))
            y_ranges.append((sy, ey))

        if device.type == "cuda":
            torch.cuda.synchronize()
        decode_start_time = time.perf_counter()

        with infer_ctx():
            for batch_start in range(0, num_patches, patch_size):
                batch_end = min(batch_start + patch_size, num_patches)
                cur_bs = batch_end - batch_start  # actual batch size

                # ---- decoder forward ----
                if device.type == "cuda":
                    with autocast():
                        patch_out = img_decoder(
                            vid_embed_s[batch_start:batch_end],
                            vid_embed_t[batch_start:batch_end],
                        )
                else:
                    patch_out = img_decoder(
                        vid_embed_s[batch_start:batch_end],
                        vid_embed_t[batch_start:batch_end],
                    )

                if patch_out.dim() == 5:
                    patch_out = patch_out[:, 0]  # assume single channel

                for local_idx in range(cur_bs):
                    global_idx = batch_start + local_idx
                    patch_fill_in = patch_out[local_idx].to(dtype=torch.float32)  # (T_patch, X_patch, Y_patch)

                    st, et = t_ranges[global_idx]
                    sx, ex = x_ranges[global_idx]
                    sy, ey = y_ranges[global_idx]

                    if overlap_kernel is None:
                        overlap_kernel = torch.ones_like(patch_fill_in)

                    weight[st:et, sx:ex, sy:ey] += overlap_kernel
                    res[st:et, sx:ex, sy:ey] += patch_fill_in
        weight[weight == 0] = 1.0
        res = res / weight

        if device.type == "cuda":
            torch.cuda.synchronize()
        decoding_time = time.perf_counter() - decode_start_time
        np_res = res.detach().cpu().numpy().astype(np.float32, copy=False)
        float_mb = np_res.nbytes / (1024.0 ** 2)
        uint16_mb = float_mb / 2.0

        print("Decoding (GPU accumulate + AMP, no I/O) time: {:.3f} s".format(decoding_time))
        print("Decoding PPS (patches per second): {:.2f}".format(num_patches / decoding_time))
        print("Decoding throughput (uint16-equivalent): {:.2f} MiB/s".format(uint16_mb / decoding_time))
        print("np_res max / min:", float(np.max(np_res)), float(np.min(np_res)))
        print()

        if hasattr(args_dict_m, "tif_min"):
            print(
                "Current min and max: {}, {}".format(
                    args_dict_m.tif_min, args_dict_m.tif_max
                )
            )
            tif_res = ndarray2tif_min_max_clip(
                np_res,
                tif_min=args_dict_m.tif_min,
                tif_max=args_dict_m.tif_max,
            )
        elif hasattr(args_dict_m, "tif_max_minus_mean"):
            print(
                "Current mean and max_minus_mean: {}, {}".format(
                    args_dict_m.tif_mean, args_dict_m.tif_max_minus_mean
                )
            )
            tif_res = ndarray2tif_mean_max_clip(
                np_res,
                tif_mean=args_dict_m.tif_mean,
                tif_max_minus_mean=args_dict_m.tif_max_minus_mean,
            )
        elif hasattr(args_dict_m, "tif_std"):
            print(
                "Current mean and std: {}, {}".format(
                    args_dict_m.tif_mean, args_dict_m.tif_std
                )
            )
            tif_res = ndarray2tif_mean_std_clip(
                np_res,
                tif_mean=args_dict_m.tif_mean,
                tif_std=args_dict_m.tif_std,
            )
        else:
            print("No tif_* metadata found in m_args, using global min-max of np_res.")
            tif_res = ndarray2tif_min_max_clip(
                np_res, tif_min=float(np_res.min()), tif_max=float(np_res.max())
            )

        if args.do_resize:
            print("tif_res.shape before resize:", tif_res.shape)
            tif_res = video_resize(tif_res, args.x_resize, args.y_resize)
            print("tif_res.shape after resize:", tif_res.shape)

        out_vid = os.path.join(cur_ckpt_store_dir, args.name + ".tif")
        tif.imwrite(out_vid, tif_res.astype(np.uint16))

        total_time = time.perf_counter() - total_start_time
        print("Reconstruction (including I/O) completes in {:.3f} s".format(total_time))
        print("Saved to:", out_vid)
        print()


if __name__ == "__main__":
    main()
