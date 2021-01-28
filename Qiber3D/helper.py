import logging
import warnings
from pathlib import Path
from sys import getsizeof
import blosc as minimizer
import numpy as np
import json
import tarfile
from io import BytesIO
from Qiber3D import config


class LookUp:
    def __init__(self, *arg, **kw):
        """
        Two way lookup dictionary
        """
        self.storage = dict(*arg)
        for key, item in kw.items():
            self.__setitem__(key, item)

    def __len__(self):
        return len(self.storage)

    def __getitem__(self, key):
        return self.storage[key]

    def __setitem__(self, key, item):
        self.storage[key] = item
        self.storage[item] = key

    def __delitem__(self, key):
        self.storage.pop(self.storage.pop(key))

    def __missing__(self, key):
        return None

    def __iter__(self):
        return self.storage.__iter__()

    def __contains__(self, item):
        return item in self.storage


class PointLookUp:

    def __init__(self, points=tuple(), places=3, convert='float'):
        """
        Two way lookup dictionary between points and point IDs.

        :param list points: list of initial points
        :param int places: round points to number of places if convert is `'float'`
        :param str convert: convert points to `'float'` or `'int'`
        """
        self.storage = dict()
        self.next_id = 0
        self.places = places
        if convert == 'float':
            self.__convert_point = self.__convert_point_float
        elif convert == 'int':
            self.__convert_point = self.__convert_point_int
        self.add_points(points)

    @property
    def __next_id(self):
        while self.next_id in self.storage:
            self.next_id += 1
        return self.next_id

    def __convert_point_float(self, point):
        return tuple([round(entry, self.places) for entry in point])

    @staticmethod
    def __convert_point_int(point):
        return tuple(point.tolist())

    def add_points(self, points):
        """
        Add multiple points at once.

        :param list points: list of points to add
        """
        for point in points:
            self.__set_clean(self.__next_id, self.__convert_point(point))

    def __len__(self):
        return len(self.storage)//2

    def __getitem__(self, key):
        if isinstance(key, (np.ndarray, list, tuple)):
            key = self.__convert_point(key)
            if key not in self.storage:
                self.__set_clean(self.__next_id, key)
        return self.storage[key]

    def __setitem__(self, key, item):
        if isinstance(key, int):
            item = self.__convert_point(item)
        elif isinstance(item, int):
            key = self.__convert_point(key)
        else:
            raise ValueError
        self.__set_clean(key, item)

    def __set_clean(self, key, item):
        self.storage[key] = item
        self.storage[item] = key

    def __delitem__(self, key):
        self.storage.pop(self.storage.pop(key))

    def __missing__(self, key):
        return None

    def __iter__(self):
        return self.storage.__iter__()

    def __contains__(self, item):
        if isinstance(item, np.ndarray):
            return self.__convert_point(item) in self.storage
        return item in self.storage


class NumpyMemoryManager:
    def __init__(self, compressor='blosclz', storage=None, meta=None):
        """
        Stores numpy arrays compressed in memory in a dictionary like structure.

        :param str compressor: a compression algorithm suporrted by `blosc <http://python-blosc.blosc.org/>`_
        :param dict storage: prefilled storage dictionary
        :param dict meta: prefilled meta dictionary

        :ivar float compression_ratio: compression ratio
        """
        if storage is None:
            self.storage = dict()
        else:
            self.storage = storage
        if meta is None:
            self.meta = dict()
        else:
            self.meta = meta
        self.cname = compressor

    def __setitem__(self, key, item):
        if type(item) != np.ndarray:
            raise TypeError(f'This class maneges only numpy.ndarray not {type(item)}')
        meta_entry = dict(
            dtype=item.dtype,
            shape=item.shape,
            nbytes=item.nbytes,
            raw_size=getsizeof(item),
            compressed_size=0,
            split=False,
        )
        if item.nbytes > 2**30:
            meta_entry['split'] = True
            meta_entry['file_name'] = []
            chunks = item.nbytes // 2**28 + 1
            storage_entry = []
            chunk_size = int(np.ceil(item.size / chunks))
            for n in range(chunks):
                storage_entry.append(minimizer.compress(item.ravel()[n*chunk_size:(n+1)*chunk_size],
                                                        cname=self.cname, typesize=item.itemsize))
                meta_entry['compressed_size'] += getsizeof(storage_entry[-1])
                meta_entry['file_name'].append(f"{key}.{n:03d}.storage")
            meta_entry['split_n'] = len(storage_entry)
        else:
            storage_entry = minimizer.compress(item.tobytes(), cname=self.cname, typesize=item.itemsize)
            meta_entry['compressed_size'] = getsizeof(storage_entry)
            meta_entry['file_name'] = f"{key}.storage"

        self.storage[key] = storage_entry
        self.meta[key] = meta_entry

    def __len__(self):
        return len(self.meta)

    def __getitem__(self, key):
        if self.meta[key]['split']:
            return np.concatenate(list((np.frombuffer(minimizer.decompress(entry), dtype=self.meta[key]['dtype']) for entry in self.storage[key])
                                       )).reshape(self.meta[key]['shape'])
        else:
            return np.frombuffer(minimizer.decompress(self.storage[key]),
                                 dtype=self.meta[key]['dtype']).reshape(self.meta[key]['shape'])

    def __delitem__(self, key):
        self.storage.pop(self.storage.pop(key))

    def __missing__(self, key):
        return None

    def __iter__(self):
        return self.storage.__iter__()

    def __contains__(self, item):
        return item in self.storage

    def __sizeof__(self):
        return sum([entry['compressed_size'] for entry in self.meta.values()])

    @classmethod
    def load(cls, in_path=None, fileobj=None):
        """
        Create a :class:`NumpyMemoryManager` from either a file or a file object.

        :param str in_path: path to load
        :param fileobj: a opened file object
        :return: :class:`NumpyMemoryManager`
        """

        with tarfile.TarFile(in_path, fileobj=fileobj, mode='r', encoding='utf-8') as save_file:
            storage = dict()
            compressor, meta_data = json.loads(save_file.extractfile('meta.json').read().decode('utf-8'))

            for key, value in meta_data.items():
                if value['split']:
                    storage[key] = []
                    for file_name in value['file_name']:
                        storage[key].append(save_file.extractfile(file_name).read())
                else:
                    storage[key] = save_file.extractfile(value['file_name']).read()

        return cls(compressor=compressor, meta=meta_data, storage=storage)

    def save(self, out_path):
        """
        Save a :class:`NumpyMemoryManager` to file.

        :param out_path: file or folder path where to save the :class:`NumpyMemoryManager`
        :type out_path: str, Path
        """

        with tarfile.TarFile(out_path, mode='w', encoding='utf-8') as save_file:
            meta_prepared = self.meta.copy()
            for key in meta_prepared:
                if hasattr(meta_prepared[key]['dtype'], 'name'):
                    meta_prepared[key]['dtype'] = meta_prepared[key]['dtype'].name
                elif hasattr(meta_prepared[key]['dtype'], '__name__'):
                    meta_prepared[key]['dtype'] = meta_prepared[key]['dtype'].__name__
                elif isinstance(meta_prepared[key]['dtype'], str):
                    pass
            meta_data = BytesIO(json.dumps([self.cname, meta_prepared]).encode('utf-8'))
            meta_data.seek(0, 2)
            meta_part = tarfile.TarInfo("meta.json")
            meta_part.size = meta_data.tell()
            meta_data.seek(0)
            save_file.addfile(meta_part, meta_data)
            for key in self.meta:
                if self.meta[key]['split']:
                    for n in range(len(self.storage[key])):
                        storage_part = tarfile.TarInfo(meta_prepared[key]['file_name'][n])
                        storage_part.size = len(self.storage[key][n])
                        save_file.addfile(storage_part, BytesIO(self.storage[key][n]))
                else:
                    storage_part = tarfile.TarInfo(meta_prepared[key]['file_name'])
                    storage_part.size = len(self.storage[key])
                    save_file.addfile(storage_part, BytesIO(self.storage[key]))

    def clear(self):
        self.storage.clear()
        self.meta.clear()

    def keys(self):
        return self.meta.keys()

    def get(self, key, default=None):
        if key in self.meta:
            return self.__getitem__(key)
        else:
            return default

    @property
    def compression_ratio(self):
        if self.meta:
            return (sum([entry['raw_size'] for entry in self.meta.values()]) /
                    sum([entry['compressed_size'] for entry in self.meta.values()]))
        else:
            return 1.0


def out_path_check(out_path, prefix, suffix, network=None, overwrite=False, logger=None):
    out_path = Path(out_path)
    if prefix is None:
        prefix = ''

    if out_path.is_dir():
        if network is not None:
            out_path = out_path / f'{prefix}{network.name}{suffix}'
        else:
            out_path = out_path / f'{prefix}{suffix}'

    if out_path.suffix == '':
        out_path = out_path.parent / (out_path.name + suffix)

    needs_unlink = False
    if out_path.is_file():
        if overwrite:
            needs_unlink = True
        else:
            if logger:
                logger.warning(f'File exist: {out_path.absolute()}')
            else:
                warnings.warn(f'File exist: {out_path.absolute()}', RuntimeWarning)
            return None, None

    return out_path, needs_unlink


def get_logger(name='Qiber3D'):
    return logging.getLogger(name)


def change_log_level(log_level, name='Qiber3D'):
    logger = logging.getLogger(name)
    logger.setLevel(log_level)


def config_logger(name='Qiber3D'):
    logger = logging.getLogger(name)
    logger.setLevel(config.log_level)
    ch = logging.StreamHandler()
    ch.setLevel(config.log_level)
    formatter = logging.Formatter('%(name)s_%(module)s [%(levelname)s] %(message)s')
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    logger.debug(f'Init Logging | {config.app_name} ({config.version})')
    return logger


def convert_to_spherical(points):
    converted_points = np.zeros_like(points, dtype=np.float64)
    if len(points.shape) == 2:
        xy = points[:, 0] ** 2 + points[:, 1] ** 2
        converted_points[:, 0] = np.sqrt(xy + points[:, 2] ** 2)  # r <= 0
        converted_points[:, 1] = np.arctan2(np.sqrt(xy), points[:, 2])
        converted_points[:, 2] = np.arctan2(points[:, 1], points[:, 0])  # -pi <= phi <= pi
    elif len(points.shape) == 1:
        xy = points[0] ** 2 + points[1] ** 2
        converted_points[0] = np.sqrt(xy + points[2] ** 2)  # r <= 0
        converted_points[1] = np.arctan2(np.sqrt(xy), points[2])
        converted_points[2] = np.arctan2(points[1], points[0])  # -pi <= phi <= pi
    else:
        raise TypeError
    return converted_points


def convert_to_cartesian(points):
    converted_points = np.zeros_like(points, dtype=np.float64)
    converted_points[:, 0] = points[:, 0] * np.sin(points[:, 1]) * np.cos(points[:, 2])
    converted_points[:, 1] = points[:, 0] * np.sin(points[:, 1]) * np.sin(points[:, 2])
    converted_points[:, 2] = points[:, 0] * np.cos(points[:, 1])
    return converted_points


def remove_direction(points, axis=2):
    converted_points = np.copy(points)
    converted_points[converted_points[:, axis] < 0] = - converted_points[converted_points[:, axis] < 0]
    return converted_points
