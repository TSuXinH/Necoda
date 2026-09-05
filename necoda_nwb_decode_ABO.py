import argparse
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

    print("==== NeCoDA dual-stream entropy ← NWB decoding ====")
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

        print("[2/3] Reading archived model and stream arrays...")
        read_network = np.array(container.compressed_network)
        read_embedding = np.array(container.compressed_embedding)
        read_embedding_index = np.array(container.compressed_embedding_index)

    print("Shapes:")
    print(f"  network          : {read_network.shape}")
    print(f"  embedding        : {read_embedding.shape}")
    print(f"  embedding_index  : {read_embedding_index.shape}")
    print()

    print("[3/3] Restoring model and stream files...")

    generate_pth_embedding(
        output_dir,
        read_embedding,
        read_embedding_index,
    )

    generate_network_embedding(
        os.path.join(output_dir, "g1"),
        read_network,
    )

    print(f"\nDecoding done. Files written to:\n{output_dir}\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--nwb-path", required=True)
    parser.add_argument("--name", default=None)
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()

    decode_from_nwb(
        nwb_path=args.nwb_path,
        name=args.name,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
