# src/ndx_necoda/container.py
import os
import io
import py7zr
import numpy as np
from pynwb.core import NWBDataInterface
from pynwb.core import register_class
from typing import List, Tuple
from hdmf.utils import docval

from pynwb import register_class
from pynwb.core import NWBDataInterface
from hdmf.utils import docval
import numpy as np


from pynwb import register_class
from pynwb.file import NWBDataInterface
from hdmf.utils import docval, get_docval, popargs
from hdmf.container import Data


from pynwb import register_class
from pynwb.file import NWBDataInterface
from hdmf.utils import docval, popargs
from hdmf.container import Container


@register_class('NecodaContainer', 'ndx-necoda')
class NecodaContainer(NWBDataInterface):

    @docval(
        {'name': 'name', 'type': str, 'doc': 'The name of this data interface'},
        {'name': 'compressed_network', 'type': 'array_data', 'doc': 'The compressed Necoda network', 'default': None},
        {'name': 'compressed_embedding', 'type': 'array_data', 'doc': 'List of compressed embeddings', 'default': None},
        {'name': 'compressed_embedding_index', 'type': ('array_data', int), 'doc': 'Length of each compressed embedding', 'default': None},
        {'name': 'original_data_size', 'type': int, 'doc': 'Original data size in bytes', 'default': None},
        {'name': 'compressed_data_size', 'type': int, 'doc': 'Compressed data size in bytes', 'default': None},
        {'name': 'documentation', 'type': str, 'doc': 'Documentation or notes', 'default': None},
    )
    def __init__(self, **kwargs):
        name = kwargs.pop('name')
        compressed_network = kwargs.pop('compressed_network', None)
        compressed_embedding = kwargs.pop('compressed_embedding', None)
        compressed_embedding_index = kwargs.pop('compressed_embedding_index', None)
        original_data_size = kwargs.pop('original_data_size', None)
        compressed_data_size = kwargs.pop('compressed_data_size', None)
        documentation = kwargs.pop('documentation', None)

        super().__init__(name=name)
        self.compressed_network = compressed_network
        self.compressed_embedding = compressed_embedding
        self.compressed_embedding_index = compressed_embedding_index
        self.original_data_size = original_data_size
        self.compressed_data_size = compressed_data_size
        self.documentation = documentation

    def create_archive_network_from_file(
            self,
            filepath: str,
            compression_level: int = 9
    ):
        """
        Reads a file from disk, compresses it, and packages it into a
        CompressedArchive NWB object.

        Args:
            name (str): A descriptive name for the archive object inside NWB.
            filepath (str): The path to the file on disk to be compressed.
            compression_level (int): The compression level to use (1-9).

        Returns:
            A new instance of the CompressedArchive class, or None on error.
        """
        if not os.path.isfile(filepath):
            print(f"Error: Source file not found at '{filepath}'")
            return None

        print(f"Creating compressed archive for file: '{filepath}'...")
        self.original_data_size = os.path.getsize(filepath)
        in_memory_archive = io.BytesIO()
        filters = [{"id": py7zr.FILTER_LZMA2, "preset": compression_level}]

        try:
            with py7zr.SevenZipFile(in_memory_archive, 'w', filters=filters) as archive:
                archive.write(filepath, os.path.basename(filepath))

            self.compressed_network = np.frombuffer(in_memory_archive.getvalue(), dtype=np.uint8)
            self.compressed_data_size = len(self.compressed_network)

            print(
                f"  - Compression successful. Original: {self.original_data_size / 1024:.3f} KB, Compressed: {self.compressed_data_size / 1024:.3f} KB")

        except Exception as e:
            print(f"An error occurred during file compression: {e}")
            return None

    def create_archive_embedding_from_file(
            self,
            base_file_path: str,
            embedding_number: int,
            epoch_num: int,
            compression_level: int = 9
    ):
        packed_compressed_embedding, index_list, compressed_size_list = \
            generate_uint8_embedding(
                base_file_path,
                embedding_number,
                epoch_num,
                compression_level,
            )
        self.compressed_embedding = packed_compressed_embedding
        self.compressed_embedding_index = ensure_array(index_list)
        self.compressed_data_size = np.sum(compressed_size_list)

    def add_doc(self, doc):
        self.documentation = doc

    def add_original_data_size(self, size):
        self.original_data_size = size


def generate_uint8_embedding(base_file_path, embedding_number, epoch_num, compression_level=9):
    compressed_embedding_list = []
    compressed_size_list = []
    for cur_embedding_id in range(embedding_number):
        cur_path = os.path.join(
            base_file_path,
            'g{}'.format(cur_embedding_id+1),
            'quant_all_{}.pth'.format(epoch_num)
        )
        in_memory_archive = io.BytesIO()
        filters = [{"id": py7zr.FILTER_LZMA2, "preset": compression_level, "dict_size": 128 << 20}]
        with py7zr.SevenZipFile(in_memory_archive, 'w', filters=filters) as archive:
            archive.write(cur_path, os.path.basename(cur_path))
        print('7z compression done')
        compressed_bytes = np.frombuffer(in_memory_archive.getvalue(), dtype=np.uint8)
        compressed_embedding_list.append(compressed_bytes)
        compressed_size_list.append(len(compressed_bytes))
        print('Current size of embedding {}: {:.3f}KB'.format(cur_embedding_id+1, len(compressed_bytes)/1024))

    packed_compressed_embedding, index_list = pack_uint8_segments(compressed_embedding_list)
    print('index_list')
    print(index_list)
    return packed_compressed_embedding, index_list, compressed_size_list


def generate_pth_embedding(output_path, packed_compressed_embedding, index_list):
    """
    Reconstructs and decompresses a list of compressed .pth files from a packed uint8 array.

    Args:
        output_path (str): Base output directory to write decompressed .pth files.
        packed_compressed_embedding (np.ndarray): Flattened uint8 array containing all compressed files.
        index_list (np.ndarray): Inclusive end indices of each compressed file segment.
        epoch_num (int): Used in filename formatting.
    """
    os.makedirs(output_path, exist_ok=True)
    compressed_embedding_list = unpack_uint8_segments(packed_compressed_embedding, index_list)
    for cur_embedding_id, compressed_bytes in enumerate(compressed_embedding_list):
        subdir = os.path.join(output_path, 'g{}'.format(cur_embedding_id+1))
        os.makedirs(subdir, exist_ok=True)
        in_memory_archive = io.BytesIO(compressed_bytes.tobytes())
        with py7zr.SevenZipFile(in_memory_archive, mode='r') as archive:
            archive.extractall(path=subdir)
        print(f"Decompressed embedding g{cur_embedding_id+1} to: {subdir}")


def generate_network_embedding(output_path, compressed_network):
    """
    Reconstructs and decompresses a list of compressed .pth files from a packed uint8 array.
    """
    os.makedirs(output_path, exist_ok=True)
    in_memory_archive = io.BytesIO(compressed_network.tobytes())
    with py7zr.SevenZipFile(in_memory_archive, mode='r') as archive:
        archive.extractall(path=output_path)
    print(f"Decompressed model_quant to: {output_path}")


def pack_uint8_segments(data_list: List[np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Flattens a list of uint8 arrays into a single flat array, along with
    an array of end indices to recover original segments.

    Args:
        data_list (List[np.ndarray]): List of 1D arrays with dtype=np.uint8.

    Returns:
        flat_data (np.ndarray): Concatenated array of all data (dtype=uint8).
        end_indices (np.ndarray): Array of end indices (inclusive) for each segment (dtype=int64).
    """
    # Type and shape check
    if not all(isinstance(x, np.ndarray) and x.dtype == np.uint8 and x.ndim == 1 for x in data_list):
        raise ValueError("All items must be 1D numpy arrays with dtype=uint8.")

    # Concatenate all arrays
    flat_data = np.concatenate(data_list)

    # Compute end indices (inclusive)
    lengths = [len(x) for x in data_list]
    end_indices = np.cumsum(lengths) - 1  # inclusive end position
    if end_indices.ndim == 0: end_indices = np.expand_dims(end_indices, axis=0)
    return flat_data, end_indices.astype(np.int64)


def unpack_uint8_segments(flat_data: np.ndarray, end_indices: np.ndarray) -> List[np.ndarray]:
    """
    Recovers original list of uint8 segments from a flat array and end indices.

    Args:
        flat_data (np.ndarray): Flat array of concatenated data (dtype=uint8).
        end_indices (np.ndarray): Array of inclusive end indices (dtype=int64).

    Returns:
        List[np.ndarray]: Reconstructed list of original segments (dtype=uint8).
    """
    if flat_data.dtype != np.uint8:
        raise ValueError("flat_data must be of dtype=uint8.")
    if end_indices.dtype != np.int64:
        raise ValueError("end_indices must be of dtype=int64.")
    if end_indices.ndim == 0: end_indices = np.expand_dims(end_indices, axis=0)
    segments = []
    start = 0
    for end in end_indices:
        segment = flat_data[start:end+1]  # include end (inclusive range)
        segments.append(segment)
        start = end + 1

    return segments


def ensure_array(x, dtype=None):
    print('type(x)')
    print(type(x))
    if isinstance(x, (np.ndarray, list, tuple)):
        return np.array(x, dtype=dtype) if dtype else np.array(x)
    else:
        return np.array([x], dtype=dtype) if dtype else np.array([x])
