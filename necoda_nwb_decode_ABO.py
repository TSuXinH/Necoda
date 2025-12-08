import os
import sys
from pathlib import Path

import numpy as np
from pynwb import NWBHDF5IO

REPO_ROOT = Path(__file__).resolve().parent
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from ndx_necoda import (
    NecodaContainer,
    generate_pth_embedding,
    generate_network_embedding,
)


def decode_from_nwb(
    nwb_path: str,
    name: str | None = None,
    output_dir: str | None = None,
) -> None:

    nwb_path = os.path.abspath(nwb_path)
    if not os.path.isfile(nwb_path):
        raise FileNotFoundError(f"NWB file not found: {nwb_path}")

    if output_dir is None:
        base = os.path.splitext(os.path.basename(nwb_path))[0]
        output_dir = os.path.join(os.getcwd(), f"output_{base}")

    os.makedirs(output_dir, exist_ok=True)

    print("==== Necoda ← NWB decoding ====")
    print(f"NWB path   : {nwb_path}")
    print(f"Output dir : {output_dir}")
    print()

    with NWBHDF5IO(nwb_path, "r") as io:
        nwbfile = io.read()
        if name is not None:
            container = nwbfile.get_acquisition(name)
            if container is None:
                raise KeyError(
                    f"No acquisition named '{name}' found in NWB file. "
                    f"Available: {list(nwbfile.acquisition.keys())}"
                )
        else:
            if len(nwbfile.acquisition) == 0:
                raise RuntimeError("No acquisitions found in NWB file.")
            name, container = next(iter(nwbfile.acquisition.items()))
            print(f"[Info] No name provided, using first acquisition: '{name}'")

        if not isinstance(container, NecodaContainer):
            print(
                f"[Warning] Acquisition '{name}' is not a NecodaContainer "
                f"(got type: {type(container)})."
            )

        print(f"[1/3] Loaded container: {name}")

        print("[2/3] Reading compressed arrays from container...")
        read_network = np.array(container.compressed_network)
        read_embedding = np.array(container.compressed_embedding)
        read_embedding_index = np.array(container.compressed_embedding_index)

    print("Shapes:")
    print(f"  network          : {read_network.shape}")
    print(f"  embedding        : {read_embedding.shape}")
    print(f"  embedding_index  : {read_embedding_index.shape}")
    print()

    print("[3/3] Regenerating files from compressed data...")

    generate_pth_embedding(
        output_dir,
        read_embedding,
        read_embedding_index,
    )

    generate_network_embedding(
        output_dir,
        read_network,
    )

    print(f"\nDecoding done. Files written to:\n{output_dir}\n")


def main():
    name = "ABO642877968"
    nwb_path = os.path.join(f"./nwb_encoded_{name}", f"{name}.nwb")
    output_dir = f"./nwb_decoded_{name}"

    try:
        decode_from_nwb(
            nwb_path=nwb_path,
            name=name,   
            output_dir=output_dir,
        )
        print("\nTest decoding finished.\n")
    except Exception:
        print("\n[ERROR] Decoding failed.")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
