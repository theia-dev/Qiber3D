import logging
from pathlib import Path
from textwrap import dedent

import networkx as nx
import numpy as np
import pims
from nd2reader import ND2Reader
from scipy import ndimage
from scipy.optimize import curve_fit
from skimage import filters

try:
    import kimimaro
except ImportError:
    kimimaro = None

import Qiber3D
from Qiber3D import config, helper


class Extractor:
    """
    Create a :class:`Qiber3D.Network` from an image stack with :meth:`get_network`. Apply filter as set in :class:`Qiber3D.config`.

    :param input_path: file path to input file
    :type input_path: str, Path
    :param channel: channel name or index
    :type channel: int, str
    :param voxel_size: size of a voxel in each axis
    :type voxel_size: list(float)

    :ivar storage: storage for extraction steps
    :vartype storage: :class:`Qiber3D.helper.NumpyMemoryManager`
    :ivar processing_data: extra information on the image processing
    :vartype processing_data: dict
    :ivar z_spacing: voxel size along z-axis
    :vartype z_spacing: float
    :ivar xy_spacing: voxel size along x/y-axis
    :vartype xy_spacing: float
    :ivar:net: generated network
    :vartype net: :class:`Qiber3D.Network`

    """

    def __init__(self, input_path, channel=None, voxel_size=None):
        self.config = config

        self.logger = helper.get_logger()

        self.core_count = self.config.core_count

        if channel is None:
            self.channel = self.config.extract.nd2_channel_name
        else:
            self.channel = channel

        self.input_path = Path(input_path)

        # Data filled with self.prepare()
        self.shape = np.zeros(3, dtype=int)
        self.positions = [None, None, None]
        self.voxel_size = voxel_size

        self.z_drop_image_stack = None
        self.z_spacing = None
        self.xy_spacing = None
        self.processing_data = {}
        self.image_stack = None
        self.segments = None
        self.net = None

        if self.config.extract.low_memory:
            self.dtype = np.float16
            self.storage = helper.NumpyMemoryManager(compressor='zstd')
        else:
            self.dtype = np.float32
            self.storage = helper.NumpyMemoryManager(compressor='blosclz')

    def get_network(self):
        """
        :return: Network from prepared image stack
        :rtype: :class:`Qiber3D.Network`
        """
        self.prepare()
        self.reconstruct()
        return self.net

    def prepare(self):
        """
        Load and prepare an image stack for reconstruction.
        """
        self.logger.info(f'Load image data from {self.input_path.absolute()}')
        self.image_stack = self.__read_raw_image()
        if self.config.extract.save_steps:
            self.storage['is_raw'] = self.image_stack
        self.__prepare_image()

    def reconstruct(self):
        """
        Reconstruct the network from the prepared image stack.
        """
        self.logger.info('reconstruct image')

        ex_data = {'xy_spacing': self.xy_spacing,
                   'z_spacing': self.z_spacing,
                   'processing_data': self.processing_data}

        if self.config.extract.use_teasar:
            self.segments = self.__teasar_reconstruct(self.image_stack,
                                                      spacing=(self.xy_spacing, self.xy_spacing, self.xy_spacing),
                                                      core_count=self.core_count,
                                                      debug=self.logger.level <= logging.DEBUG)
            network_data = {
                'path': self.input_path,
                'name': self.input_path.with_suffix('').name,
                'segments': self.segments
            }
            self.net = Qiber3D.Network(network_data)
        else:
            self.net = Qiber3D.Reconstruct.get_network(self.image_stack, scale=self.xy_spacing,
                                                       input_path=self.input_path,
                                                       sliver_threshold=self.config.extract.thinning.sliver_threshold,
                                                       voxel_per_point=self.config.extract.thinning.voxel_per_point,
                                                       low_memory=self.config.extract.low_memory,
                                                       distance_voxel_overlap=self.config.extract.thinning.distance_voxel_overlap
                                                       )
        self.net.extractor_data = ex_data
        self.net.extractor_steps = self.storage

    def __show_step(self, name, binary=False, spacing=None):
        if self.logger.level <= logging.DEBUG:
            if spacing is None:
                spacing = (self.xy_spacing, self.xy_spacing, self.z_spacing)
            Qiber3D.Render.show_3d_image(self.image_stack, name, binary=binary, spacing=spacing)

    def __read_raw_image(self):
        images_selected = None
        if self.input_path.suffix == '.nd2':
            images = ND2Reader(str(self.input_path.absolute()))
            self.shape[:2] = images.frame_shape
            self.shape[2] = images.sizes['z']
            self.positions[2] = np.array(images.metadata['z_coordinates'])
            self.positions[2] += - min(self.positions[2])
            for n in (0, 1):
                self.positions[n] = np.linspace(0, images.metadata['pixel_microns']*self.shape[n],
                                                num=self.shape[n], endpoint=False)
            if 'c' in images.default_coords:
                if type(self.channel) == int:
                    images.default_coords['c'] = self.channel
                else:
                    images.default_coords['c'] = images.metadata['channels'].index(self.channel)

            self.z_spacing = np.average(np.diff(self.positions[2]))
            self.xy_spacing = images.metadata['pixel_microns']
        else:
            images = pims.open(str(self.input_path.absolute()))
            if hasattr(images, '_tiff') and self.channel is not None:

                self.shape[:2] = images.frame_shape
                self.shape[2] = images._tiff.shape[0]
                if len(images._tiff.shape) == 4:
                    images_selected = images[self.channel::images._tiff.shape[1]]
                else:
                    images_selected = images

            else:
                self.shape[:2] = images.frame_shape
                self.shape[2] = len(images)
            for n in range(3):
                self.positions[n] = np.linspace(0, self.voxel_size[n]*self.shape[n],
                                                num=self.shape[n], endpoint=False)
            self.z_spacing = self.voxel_size[2]
            self.xy_spacing = self.voxel_size[0]

        # scale all images from 0 to 1 and store it at the selected resolution
        self.logger.info(f'Image voxel size: [{self.xy_spacing:.3f},{self.xy_spacing:.3f},{self.z_spacing:.3f}]')
        raw_image = np.zeros((self.shape[2], self.shape[0], self.shape[1]), dtype=self.dtype)
        if images_selected:
            raw_image[:] = (images_selected - np.min(images_selected)) / (np.max(images_selected) - np.min(images_selected))
        else:
            raw_image[:] = (images - np.min(images))/(np.max(images)-np.min(images))
        if self.config.extract.invert:
            raw_image = 1.0 - raw_image
        images.close()
        return raw_image

    def __prepare_image(self):
        self.__show_step('Original')

        if self.config.extract.median.apply:
            self.logger.info('Median Filter (despeckle)')
            if self.config.extract.low_memory:
                self.image_stack = self.filter_median(self.image_stack.astype(np.float32)).astype(self.dtype)
            else:
                self.image_stack = self.filter_median(self.image_stack)
            if self.config.extract.save_steps:
                self.storage['is_median'] = self.image_stack
            self.__show_step('Median _filter')

        if self.config.extract.z_drop.apply:
            self.logger.info('Z-Drop correction')
            self.image_stack = self.filter_z_drop_correction(self.image_stack)
            if self.config.extract.save_steps:
                self.storage['is_z_drop'] = self.image_stack
            self.__show_step('Z-drop correction')

        self.logger.info('Resample image to cubic voxels')
        z_change = self.z_spacing / self.xy_spacing
        if self.config.extract.low_memory:
            self.image_stack = self.filter_resample(self.image_stack.astype(np.float32), z_change=z_change,
                                                    target=self.config.extract.isotropic_resampling.target).astype(self.dtype)
        else:
            self.image_stack = self.filter_resample(self.image_stack, z_change=z_change,
                                                    target=self.config.extract.isotropic_resampling.target)
        if self.config.extract.isotropic_resampling.target == 'xy':
            self.xy_spacing = self.z_spacing
        self.__show_step('Cubic resampling', spacing=[self.xy_spacing]*3)

        if self.config.extract.smooth.apply:
            self.logger.info('Apply gaussian filter')
            if self.config.extract.low_memory:
                self.image_stack = self.filter_smooth(self.image_stack.astype(np.float32)).astype(self.dtype)
            else:
                self.image_stack = self.filter_smooth(self.image_stack)
            if self.config.extract.save_steps:
                self.storage['is_smooth'] = self.image_stack
            self.__show_step('Gaussian filter', spacing=[self.xy_spacing] * 3)

        self.logger.info('Generate binary representation')
        self.image_stack, used_threshold, requested_threshold = self.binary_representation(self.image_stack)
        self.logger.info(f'Binary representation used a threshold of {used_threshold:.1f}% ({requested_threshold})')
        self.__show_step('Binary representation', binary=True, spacing=[self.xy_spacing] * 3)

        if self.config.extract.morph.apply:
            self.logger.info('Morph binary representation')
            self.image_stack = self.filter_morph(self.image_stack, voxel_spacing=self.xy_spacing)
            self.__show_step('Morphed binary (final)', binary=True, spacing=[self.xy_spacing] * 3)

        if self.config.extract.save_steps:
            self.storage['is_final'] = self.image_stack

    @staticmethod
    def __teasar_reconstruct(image, spacing, core_count=0, debug=False):
        if kimimaro is None:
            raise ImportError("kimimaro is not installed (pip install -U kimimaro)")
            return
        label_im, nb_labels = ndimage.label(image)
        teasar_params = {}
        for key in ['scale', 'const', 'pdrf_exponent', 'pdrf_scale',
                    'soma_detection_threshold', 'soma_acceptance_threshold',
                    'soma_invalidation_scale', 'soma_invalidation_const', 'max_paths']:
            teasar_params[key] = getattr(config.extract.teasar, key)
        skeleton = kimimaro.skeletonize(
            label_im,
            teasar_params=teasar_params,
            dust_threshold=config.extract.teasar.dust_threshold // np.prod(spacing),  # skip connected components with fewer than this many voxels
            anisotropy=spacing,  # default True
            fix_branching=True,  # default True
            fix_borders=True,  # default True
            fill_holes=False,  # default False
            fix_avocados=False,  # default False
            progress=debug,  # default False, show progress bar
            parallel=core_count,  # <= 0 all cpu, 1 single process, 2+ multiprocess
            parallel_chunk_size=50,  # how many skeletons to process before updating progress bar
        )

        segment_data = {}
        seg_id = 0
        for network in skeleton.values():
            segments = []
            start_points = None
            stop_points = []
            network_graph = nx.Graph()
            network_graph.add_edges_from(network.edges)
            for node in network_graph:
                if len(network_graph.adj[node]) == 1:
                    start_points = [(node,)]
                    break
            if start_points is None:
                start_points = [(network_graph.nodes[0],)]

            while start_points:
                start = start_points.pop()
                if len(start) == 1:
                    new_segment = [start[0]]
                    start = start[0]
                else:
                    if not network_graph.has_edge(start[0], start[1]):
                        continue
                    new_segment = [start[0], start[1]]
                    network_graph.remove_edge(start[0], start[1])
                    start = start[1]
                for f, t in nx.dfs_successors(network_graph, start).items():
                    if len(t) == 1:
                        new_segment.append(t[0])
                        network_graph.remove_edge(f, t[0])
                        if t[0] in stop_points:
                            break
                    elif len(t) > 1:
                        for paths in t:
                            start_points.append((f, paths))
                        stop_points.append(f)
                        break
                segments.append(new_segment)

            for seg in segments:
                points = np.array([(round(x, 4), round(y, 4), round(z, 4)) for (z, y, x) in network.vertices[seg]])
                segment_data[seg_id] = dict(
                    points=points,
                    radius=network.radius[seg],
                    seg_id=seg_id
                )
                seg_id += 1
        return segment_data

    @staticmethod
    def filter_morph(image, iterations=None, remove_vol=None, voxel_spacing=1):
        if iterations is None:
            iterations = config.extract.morph.iterations
        if remove_vol is None:
            remove_vol = config.extract.morph.remove_vol
        remove_vol /= voxel_spacing ** 3
        image = ndimage.binary_dilation(image, iterations=iterations)
        image = ndimage.binary_erosion(image, iterations=iterations)
        label_im, nb_labels = ndimage.label(image)
        sizes = ndimage.sum(image, label_im, range(nb_labels + 1))
        mask = sizes > remove_vol
        image = mask[label_im]
        return image

    @staticmethod
    def filter_resample(image, z_change=1.0, target='z'):
        if round(z_change, 3) != 1.0:
            if target == 'z':
                return ndimage.zoom(image, (z_change, 1, 1), order=1)
            elif target == 'xy':
                return ndimage.zoom(image, (1, 1/z_change, 1/z_change), order=1)
        else:
            return image

    @staticmethod
    def filter_smooth(image, sigma=None, truncate=None):
        if sigma is None:
            sigma = config.extract.smooth.sigma
        if truncate is None:
            truncate = config.extract.smooth.truncate
        return ndimage.gaussian_filter(image, sigma, mode='mirror', truncate=truncate)

    @staticmethod
    def filter_median(image, size=None, footprint=None):
        if size is None:
            size = config.extract.median.size
        if footprint is None:
            footprint = config.extract.median.footprint
        if footprint is not None:
            return ndimage.median_filter(image, footprint=footprint)
        else:
            return ndimage.median_filter(image, size=size)

    @staticmethod
    def binary_representation(image, threshold=None):
        if threshold is None:
            threshold = config.extract.binary.threshold
        if threshold is None:
            threshold = 'otsu'

        if type(threshold) == str:
            threshold = threshold.lower().strip()
            if threshold == 'otsu':
                applied_threshold = filters.threshold_otsu(image)
            elif threshold == 'li':
                applied_threshold = filters.threshold_li(image)
            elif threshold == 'isodata':
                applied_threshold = filters.threshold_isodata(image)
            elif threshold == 'mean':
                applied_threshold = filters.threshold_mean(image)
            elif threshold == 'minimum':
                applied_threshold = filters.threshold_minimum(image)
            elif threshold == 'triangle':
                applied_threshold = filters.threshold_triangle(image)
            elif threshold == 'yen':
                applied_threshold = filters.threshold_yen(image)
            else:
                applied_threshold = filters.threshold_otsu(image)
                threshold = f"could not find {threshold} using otsu"
        else:
            applied_threshold = threshold / 100 * np.max(image)
            threshold = f"direct"
        return image >= applied_threshold, (applied_threshold/np.max(image))*100, threshold

    @staticmethod
    def _z_drop_func(x, a, b):
        return a * np.exp(b * x)

    def filter_z_drop_correction(self, image):
        mean_intensities = np.average(image, axis=(1, 2))
        x = (self.positions[2] / max(self.positions[2])).astype(self.dtype)
        parameters_optimized, p_conv = curve_fit(self._z_drop_func, x, mean_intensities)
        z_scale = self._z_drop_func(x, np.average(mean_intensities)/parameters_optimized[0], -parameters_optimized[1])
        image = image * z_scale[:, np.newaxis, np.newaxis]
        image /= np.max(image)  # scale to 1.0
        self.processing_data['z_drop'] = {'parameter': parameters_optimized, 'x': x,
                                          'z_fit': self._z_drop_func(x, *parameters_optimized),
                                          'y': mean_intensities, 'y_corrected':  mean_intensities*z_scale}

        return image

    def __str__(self):
        info = f"""\
        Input file: {self.input_path.name}
          Number of segments: {len(self.segments)}
          """
        return dedent(info)

    def __repr__(self):
        return f"Extractor for '{self.input_path.name}'"



