import os
import sys
import uuid
import datetime
from pathlib import Path

from pynwb import NWBHDF5IO, NWBFile

REPO_ROOT = Path(__file__).resolve().parent
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from ndx_necoda import NecodaContainer


def encode_to_nwb(
    name: str,
    base_path: str,
    epoch_num: int,
    embedding_number: int,
    original_data_size: int,
    output_dir: str | None = None,
) -> str:

    if output_dir is None:
        output_dir = f"./output_{name}"

    output_dir = os.path.abspath(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    nwb_path = os.path.join(output_dir, f"{name}.nwb")

    print("==== Necoda → NWB encoding ====")
    print(f"Name              : {name}")
    print(f"Base path         : {base_path}")
    print(f"Epoch number      : {epoch_num}")
    print(f"Embedding number  : {embedding_number}")
    print(f"Original data size: {original_data_size} bytes")
    print(f"Output dir        : {output_dir}")
    print(f"NWB file          : {nwb_path}")
    print()

    container = NecodaContainer(name=name)

    model_pth = os.path.join(base_path, f"model_quant_{epoch_num}.pth")
    if not os.path.isfile(model_pth):
        raise FileNotFoundError(f"Quantized model file not found: {model_pth}")

    print(f"[1/3] Archiving quantized network from: {model_pth}")
    container.create_archive_network_from_file(filepath=model_pth)

    container.add_doc(f"Compression of {name}")
    container.add_original_data_size(original_data_size)

    print(f"[2/3] Archiving embeddings from base path: {base_path}")
    container.create_archive_embedding_from_file(
        base_file_path=base_path,
        embedding_number=embedding_number,
        epoch_num=epoch_num,
    )

    print()
    print(f"Total raw size       : {container.original_data_size / 1024**3:.3f} GB")
    print(f"Total compressed size: {container.compressed_data_size / 1024**2:.3f} MB")
    print()

    print("[3/3] Writing NWB file...")
    nwbfile = NWBFile(
        session_description=f"Necoda compression of {name}",
        identifier=str(uuid.uuid4()),
        session_start_time=datetime.datetime.now(datetime.timezone.utc),
    )
    nwbfile.add_acquisition(container)

    with NWBHDF5IO(nwb_path, "w") as io:
        io.write(nwbfile)

    print(f"\nEncoding done. NWB saved to:\n{nwb_path}\n")
    return nwb_path


def main():

    name = "ABO642877968"
    epoch_num = 60
    embedding_number = 2
    original_data_size = int(3072973 * 2 * 1024)  # calculated using raw data

    # Replace with the true path
    base_path = "./result"
    output_dir = f"./nwb_encoded_{name}"

    try:
        encode_to_nwb(
            name=name,
            base_path=base_path,
            epoch_num=epoch_num,
            embedding_number=embedding_number,
            original_data_size=original_data_size,
            output_dir=output_dir,
        )
        print("\nTest encoding finished.\n")
    except Exception as e:
        print("\n[ERROR] Encoding failed.")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
