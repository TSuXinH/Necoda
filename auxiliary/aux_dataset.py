import random

import numpy as np

import tifffile as tif

import torch

from torch.utils.data import Dataset

from .process_tif import apply_normalization

def random_transform(array, p_trans):
    if p_trans == 0:
        return array
    if p_trans == 1:
        return np.rot90(array, k=1, axes=(1, 2))
    if p_trans == 2:
        return np.rot90(array, k=2, axes=(1, 2))
    if p_trans == 3:
        return np.rot90(array, k=3, axes=(1, 2))
    if p_trans == 4:
        return array[:, :, ::-1]
    if p_trans == 5:
        return np.rot90(array[:, :, ::-1], k=1, axes=(1, 2))
    if p_trans == 6:
        return np.rot90(array[:, :, ::-1], k=2, axes=(1, 2))
    if p_trans == 7:
        return np.rot90(array[:, :, ::-1], k=3, axes=(1, 2))
    raise ValueError(f"Unsupported transform id: {p_trans}")

def _set_legacy_norm_attributes(dataset, norm_stats):
    """
    Keep legacy attributes used by older training/reconstruction scripts.
    """
    mode = norm_stats["mode"]

    if mode in ("min_max", "robust_min_max"):
        dataset.tif_min = float(norm_stats["min"])
        dataset.tif_max = float(norm_stats["max"])

    elif mode == "mean":
        dataset.tif_mean = float(norm_stats["mean"])

    elif mode == "mean_std":
        dataset.tif_mean = float(norm_stats["mean"])
        dataset.tif_std = float(norm_stats["std"])

    elif mode == "mean_max":
        dataset.tif_mean = float(norm_stats["mean"])
        dataset.tif_max_minus_mean = float(
            norm_stats["max_minus_mean"]
        )

def _load_and_normalize(
    data_path,
    pre_norm,
    norm_stats,
    robust_low,
    robust_high,
):
    tif_arr = tif.imread(data_path)
    neural_data, used_stats = apply_normalization(
        tif_arr,
        pre_norm=pre_norm,
        norm_stats=norm_stats,
        robust_low=robust_low,
        robust_high=robust_high,
    )

    print("Normalization stats:", used_stats)
    print("Normalized shape:", neural_data.shape)
    print(
        "Normalized min/max/mean/std:",
        float(neural_data.min()),
        float(neural_data.max()),
        float(neural_data.mean()),
        float(neural_data.std()),
    )

    return neural_data, used_stats

def _axis_starts(total_size, patch_size, step):
    """
    Generate patch starts and always cover the final boundary.

    The old implementation silently omitted the final 48 frames for
    T=6000, patch_t=128 and gap_t=64.
    """
    if patch_size > total_size:
        raise ValueError(
            f"patch_size={patch_size} exceeds total_size={total_size}"
        )
    if step <= 0:
        raise ValueError(f"step must be positive, got {step}")

    starts = list(range(0, total_size - patch_size + 1, step))
    final_start = total_size - patch_size

    if not starts or starts[-1] != final_start:
        starts.append(final_start)

    return starts

class DatasetTifPatchTrainWithPadding(Dataset):
    def __init__(
        self,
        data_path,
        patch_x=128,
        patch_t=128,
        gap_x=64,
        gap_t=64,
        interp_x=4,
        interp_t=4,
        apply_aug=False,
        apply_sampling=False,
        pre_norm="robust_min_max",
        norm_stats=None,
        robust_low=0.001,
        robust_high=99.99,
    ):
        super().__init__()

        self.neural_data, self.norm_stats = _load_and_normalize(
            data_path,
            pre_norm,
            norm_stats,
            robust_low,
            robust_high,
        )
        _set_legacy_norm_attributes(self, self.norm_stats)

        self.x = self.neural_data.shape[1]
        self.y = self.neural_data.shape[2]
        self.t = self.neural_data.shape[0]
        self.patch_x = patch_x
        self.patch_y = patch_x
        self.patch_t = patch_t
        self.interp_x = interp_x
        self.interp_y = interp_x
        self.interp_t = interp_t
        self.apply_aug = apply_aug
        self.apply_sampling = apply_sampling

        self.coordinate_list = create_overlap_patch_info_train(
            patch_x,
            gap_x,
            self.x,
            patch_t,
            gap_t,
            self.t,
            self.y,
        )

        print(
            "Finish creating padded training dataset. Patches:",
            len(self.coordinate_list),
        )

    def __len__(self):
        return len(self.coordinate_list)

    def __getitem__(self, item):
        t0, x0, y0 = self.coordinate_list[item]

        start_t, end_t = get_interp_coord(
            t0,
            self.t,
            self.patch_t,
            self.interp_t,
        )
        start_x, end_x = get_interp_coord(
            x0,
            self.x,
            self.patch_x,
            self.interp_x,
        )
        start_y, end_y = get_interp_coord(
            y0,
            self.y,
            self.patch_y,
            self.interp_y,
        )

        stack = self.neural_data[
            start_t:end_t,
            start_x:end_x,
            start_y:end_y,
        ]

        raw, target = (
            hierarchical_data_generation(stack)
            if self.apply_sampling
            else (stack.copy(), stack.copy())
        )

        if self.apply_aug:
            p_trans = random.randrange(8)
            raw = random_transform(raw, p_trans)
            target = random_transform(target, p_trans)

        return {
            "patch": torch.from_numpy(
                np.ascontiguousarray(raw[None], dtype=np.float32)
            ),
            "target": torch.from_numpy(
                np.ascontiguousarray(target[None], dtype=np.float32)
            ),
            "patch_id": item,
        }

class DatasetTifPatchTest(Dataset):
    def __init__(
        self,
        data_path,
        patch_x=128,
        patch_t=128,
        interp_x=4,
        interp_t=4,
        pre_norm="robust_min_max",
        norm_stats=None,
        robust_low=0.001,
        robust_high=99.99,
    ):
        super().__init__()

        self.neural_data, self.norm_stats = _load_and_normalize(
            data_path,
            pre_norm,
            norm_stats,
            robust_low,
            robust_high,
        )
        _set_legacy_norm_attributes(self, self.norm_stats)

        self.x = self.neural_data.shape[1]
        self.y = self.neural_data.shape[2]
        self.t = self.neural_data.shape[0]
        self.patch_x = patch_x
        self.patch_y = patch_x
        self.patch_t = patch_t
        self.interp_x = interp_x
        self.interp_y = interp_x
        self.interp_t = interp_t

        self.coordinate_list = create_overlap_patch_info_test(
            patch_x,
            self.x,
            patch_t,
            self.t,
            self.y,
        )

        print(
            "Finish creating padded test dataset. Patches:",
            len(self.coordinate_list),
        )

    def __len__(self):
        return len(self.coordinate_list)

    def __getitem__(self, item):
        t0, x0, y0 = self.coordinate_list[item]

        start_t, end_t = get_interp_coord(
            t0,
            self.t,
            self.patch_t,
            self.interp_t,
        )
        start_x, end_x = get_interp_coord(
            x0,
            self.x,
            self.patch_x,
            self.interp_x,
        )
        start_y, end_y = get_interp_coord(
            y0,
            self.y,
            self.patch_y,
            self.interp_y,
        )

        stack = self.neural_data[
            start_t:end_t,
            start_x:end_x,
            start_y:end_y,
        ]

        return {
            "patch": torch.from_numpy(
                np.ascontiguousarray(stack[None], dtype=np.float32)
            ),
            "target": torch.from_numpy(
                np.ascontiguousarray(stack[None], dtype=np.float32)
            ),
            "patch_id": item,
        }

def create_overlap_patch_info_train(
    patch_x,
    gap_x,
    whole_x,
    patch_t,
    gap_t,
    whole_t,
    whole_y=0,
):
    patch_y = patch_x
    gap_y = gap_x
    whole_y = whole_x if whole_y == 0 else whole_y

    t_starts = _axis_starts(whole_t, patch_t, gap_t)
    x_starts = _axis_starts(whole_x, patch_x, gap_x)
    y_starts = _axis_starts(whole_y, patch_y, gap_y)

    # Keep the same x -> y -> t ordering used by the legacy code.
    return [
        [t0, x0, y0]
        for x0 in x_starts
        for y0 in y_starts
        for t0 in t_starts
    ]

def create_overlap_patch_info_test(
    patch_x,
    whole_x,
    patch_t,
    whole_t,
    whole_y=0,
):
    patch_y = patch_x
    whole_y = whole_x if whole_y == 0 else whole_y

    t_starts = _axis_starts(whole_t, patch_t, patch_t)
    x_starts = _axis_starts(whole_x, patch_x, patch_x)
    y_starts = _axis_starts(whole_y, patch_y, patch_y)

    return [
        [t0, x0, y0]
        for x0 in x_starts
        for y0 in y_starts
        for t0 in t_starts
    ]

def get_interp_coord(
    cur_start_coord,
    total_size,
    patch_size,
    interp_length,
):
    if cur_start_coord < interp_length:
        start_p = cur_start_coord
        end_p = cur_start_coord + patch_size + 2 * interp_length
    elif (
        cur_start_coord + patch_size + interp_length
        > total_size
    ):
        end_p = cur_start_coord + patch_size
        start_p = cur_start_coord - 2 * interp_length
    else:
        start_p = cur_start_coord - interp_length
        end_p = (
            cur_start_coord + patch_size + interp_length
        )

    if start_p < 0 or end_p > total_size:
        raise ValueError(
            f"Invalid interpolated range [{start_p}, {end_p}) "
            f"for total_size={total_size}"
        )

    return start_p, end_p

def random_pick(dim):
    return [np.random.randint(0, 2) for _ in range(dim)]

def spatiotemporal_sampling(stack):
    raw_rand_list = random_pick(3)
    target_rand_list = random_pick(3)

    while raw_rand_list == target_rand_list:
        target_rand_list = random_pick(3)

    raw = stack[
        raw_rand_list[0]::2,
        raw_rand_list[1]::2,
        raw_rand_list[2]::2,
    ]
    target = stack[
        target_rand_list[0]::2,
        target_rand_list[1]::2,
        target_rand_list[2]::2,
    ]
    return raw, target

def hierarchical_data_generation(stack):
    return spatiotemporal_sampling(stack)
