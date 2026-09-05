# src/ndx_necoda/container.py
import io
import os
from typing import List, Tuple

import numpy as np
import py7zr
from hdmf.utils import docval
from pynwb import register_class
from pynwb.file import NWBDataInterface


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
        network_size = 0 if self.compressed_network is None else len(self.compressed_network)
        self.compressed_data_size = int(network_size + np.sum(compressed_size_list))

    def add_doc(self, doc):
        self.documentation = doc

    def add_original_data_size(self, size):
        self.original_data_size = size


def generate_uint8_embedding(base_file_path, embedding_number, epoch_num, compression_level=9):
    compressed_embedding_list = []
    compressed_size_list = []
    for cur_embedding_id in range(embedding_number):
        dataset_dir = os.path.join(base_file_path, 'g{}'.format(cur_embedding_id+1))
        stream_path = os.path.join(dataset_dir, 'c_dict_{}.pth'.format(epoch_num))
        args_path = os.path.join(dataset_dir, 'args.pth')
        for required_path in (stream_path, args_path):
            if not os.path.isfile(required_path):
                raise FileNotFoundError(required_path)
        in_memory_archive = io.BytesIO()
        filters = [{"id": py7zr.FILTER_LZMA2, "preset": compression_level, "dict_size": 128 << 20}]
        with py7zr.SevenZipFile(in_memory_archive, 'w', filters=filters) as archive:
            archive.write(stream_path, os.path.basename(stream_path))
            archive.write(args_path, os.path.basename(args_path))
        print('7z compression done')
        compressed_bytes = np.frombuffer(in_memory_archive.getvalue(), dtype=np.uint8)
        compressed_embedding_list.append(compressed_bytes)
        compressed_size_list.append(len(compressed_bytes))
        print('Current size of stream bundle {}: {:.3f}KB'.format(cur_embedding_id+1, len(compressed_bytes)/1024))

    packed_compressed_embedding, index_list = pack_uint8_segments(compressed_embedding_list)
    print('index_list')
    print(index_list)
    return packed_compressed_embedding, index_list, compressed_size_list


def generate_pth_embedding(output_path, packed_compressed_embedding, index_list):
    os.makedirs(output_path, exist_ok=True)
    compressed_embedding_list = unpack_uint8_segments(packed_compressed_embedding, index_list)
    for cur_embedding_id, compressed_bytes in enumerate(compressed_embedding_list):
        subdir = os.path.join(output_path, 'g{}'.format(cur_embedding_id+1))
        os.makedirs(subdir, exist_ok=True)
        in_memory_archive = io.BytesIO(compressed_bytes.tobytes())
        with py7zr.SevenZipFile(in_memory_archive, mode='r') as archive:
            archive.extractall(path=subdir)
        print(f"Restored stream bundle g{cur_embedding_id+1} to: {subdir}")


def generate_network_embedding(output_path, compressed_network):

    os.makedirs(output_path, exist_ok=True)
    in_memory_archive = io.BytesIO(compressed_network.tobytes())
    with py7zr.SevenZipFile(in_memory_archive, mode='r') as archive:
        archive.extractall(path=output_path)
    print(f"Restored model checkpoint to: {output_path}")


def pack_uint8_segments(data_list: List[np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
    if not all(isinstance(x, np.ndarray) and x.dtype == np.uint8 and x.ndim == 1 for x in data_list):
        raise ValueError("All items must be 1D numpy arrays with dtype=uint8.")

    flat_data = np.concatenate(data_list)

    lengths = [len(x) for x in data_list]
    end_indices = np.cumsum(lengths) - 1  # inclusive end position
    if end_indices.ndim == 0: end_indices = np.expand_dims(end_indices, axis=0)
    return flat_data, end_indices.astype(np.int64)


def unpack_uint8_segments(flat_data: np.ndarray, end_indices: np.ndarray) -> List[np.ndarray]:
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
