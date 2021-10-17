import random
import os
import shutil
import string
import subprocess
import time
from functools import partial
from multiprocessing import Pool, cpu_count
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import tifffile as tif
import vedo
import vtk
from PIL import Image
from matplotlib import cm
from scipy import ndimage
from skimage import filters
from vtk.util.numpy_support import vtk_to_numpy

from Qiber3D import helper, config

if config.render.notebook:
    from tqdm.notebook import tqdm
else:
    from tqdm import tqdm


class Render:
    """
    Generate 3D representations

    :param network:
    :type network: :class:`Qiber3D.Network`

    :ivar storage: storage for rasterized data
    :vartype storage: :class:`Qiber3D.helper.NumpyMemoryManager`
    :ivar raster: rasterized representation of the network
    :vartype raster: np.ndarray
    :ivar raster_resolution: voxel per length unit
    :vartype raster_resolution: float
    :ivar raster_offset: distance between raster image stack origin and network orgin
    :vartype raster_offset: tuple(float)

    """

    logger = helper.get_logger()

    def __init__(self, network):
        self.network = network

        self.work_dir_prefix = f'{config.app_name}_rendering_'
        self.storage = helper.NumpyMemoryManager()
        self.raster_resolution = None
        self.raster_offset = None

        if config.log_level <= 10:
            self.terminal_out = None
        else:
            self.terminal_out = subprocess.DEVNULL
        self.size_sort = np.argsort(self.network.bbox_size)

    def __color_flat(self, color):
        for segment in self.network.segment.values():
            if color:
                segment._color = color
            else:
                segment._color = config.render.color

    def __color_fiber_length(self, color_map):
        max_length = 0
        for fiber in self.network.clustered_segments:
            max_length = max(max_length, sum([self.network.segment[sid].length for sid in fiber]))

        cmp = cm.get_cmap(color_map)
        for fiber in self.network.fiber.values():
            for sid in fiber.segment:
                self.network.segment[sid]._color = cmp(fiber.length / max_length)[:3]

    def __color_fiber_volume(self, color_map):
        max_volume= 0
        for fiber in self.network.clustered_segments:
            max_volume = max(max_volume, sum([self.network.segment[sid].volume for sid in fiber]))

        cmp = cm.get_cmap(color_map)
        for fiber in self.network.fiber.values():
            for sid in fiber.segment:
                self.network.segment[sid]._color = cmp(fiber.volume / max_volume)[:3]

    def __color_fiber_segment_ratio(self, color_map):
        max_segment_count = 0
        for fiber in self.network.clustered_segments:
            max_segment_count = max(max_segment_count, len(fiber))

        for sid in self.network.separate_segments:
            self.network.segment[sid]._color = [0.5, 0.5, 0.5]

        cmp = cm.get_cmap(color_map)
        for fiber in self.network.clustered_segments:
            for sid in fiber:
                self.network.segment[sid]._color = cmp(len(fiber) / max_segment_count)[:3]

    def __color_fiber(self, color_map):
        for sid in self.network.separate_segments:
            self.network.segment[sid]._color = [0.5, 0.5, 0.5]

        cmp = cm.get_cmap(color_map)
        for n, fiber in enumerate(self.network.clustered_segments):
            for sid in fiber:
                self.network.segment[sid]._color = cmp(n / len(self.network.clustered_segments))[:3]

    def __color_segment(self, color_map):
        cmp = cm.get_cmap(color_map)
        random.seed('segment_seed')
        sample = random.sample(list(self.network.segment), len(self.network.segment))
        for n, sid in enumerate(sample):
            self.network.segment[sid]._color = cmp(n / len(self.network.segment))[:3]

    def __color_segment_length(self, color_map):
        max_length = max([segment.length for segment in self.network.segment.values()])
        cmp = cm.get_cmap(color_map)
        for segment in self.network.segment.values():
            segment._color = cmp(segment.length / max_length)[:3]

    def __color_segment_volume(self, color_map):
        max_volume = max([segment.volume for segment in self.network.segment.values()])
        cmp = cm.get_cmap(color_map)
        for segment in self.network.segment.values():
            segment._color = cmp(segment.volume / max_volume)[:3]

    def __color(self, color_mode, color, color_map):
        if color_mode == 'flat':
            self.__color_flat(color)
        elif color_mode == 'fiber':
            self.__color_fiber(color_map)
        elif color_mode == 'fiber_length':
            self.__color_fiber_length(color_map)
        elif color_mode == 'fiber_volume':
            self.__color_fiber_volume(color_map)
        elif color_mode == 'segment':
            self.__color_segment(color_map)
        elif color_mode == 'segment_length':
            self.__color_segment_length(color_map)
        elif color_mode == 'segment_volume':
            self.__color_segment_volume(color_map)
        elif color_mode == 'fiber_segment_ratio':
            self.__color_fiber_segment_ratio(color_map)

    def __set_up_objects(self, color_mode=None, color=None, color_map=None, object_type=None, segment_list=None, raster_prepare=False):
        if not raster_prepare:
            self.__color(color_mode, color, color_map)

        if segment_list is None:
            render_segments = self.network.segment.values()
        else:
            render_segments = [self.network.segment[sid] for sid in segment_list]

        if object_type == 'line':
            obj_list = [vedo.Line(seg.point, c=seg._color, lw=2) for seg in render_segments]
        elif object_type == 'mixed':
            rastered = self.raster
            obj_list = [vedo.Volume(rastered, spacing=[1.0/self.raster_resolution]*3, origin=-self.raster_offset/self.raster_resolution)]
            obj_list += [vedo.Line(seg.point, c=(0, 0, 0)) for seg in render_segments]

        else:
            obj_list = [vedo.Tube(seg.point, r=seg.radius, c=seg._color) for seg in render_segments]
        if raster_prepare:
            return obj_list, [(seg.point, seg.radius) for seg in render_segments]
        else:
            return obj_list

    def show(self, color_mode='flat', color_map='jet', color=None, object_type=None, segment_list=None):
        """
        Visualize a network interactively.

        :param str color_mode: sets the way to color the network
            choose one of ['flat', 'fiber', 'fiber_length', 'fiber_volume', 'segment',
            'segment_length', 'segment_volume', 'fiber_segment_ratio']
        :param str color_map: name of a matplotlib colormap
        :param tuple(float) color: color if color_mode is `'flat'`
        :param str object_type: when set to `'line'` render center line
        :param tuple segment_list: limit the visualisation to certain segment (use sid)
        """
        object_list = self.__set_up_objects(color_mode, color, color_map, object_type, segment_list)
        vedo.settings.useParallelProjection = True
        if config.render.notebook:
            helper.notebook_display_backend()
            tube_text = []
        else:
            tube_text = [vedo.Text2D(self.network.name, c='black')]
        window = vedo.show(object_list + tube_text, axes=1)
        if config.render.notebook:
            return window
        window.close()

    def compare(self, color_mode='flat', color_map='jet', color=None, object_type=None, segment_list=None):
        """
        Visualize extraction steps (original image, z-drop image, binary image, reconstruction) at once.
        The parameter influence just the reconstructed network.

        :param str color_mode: sets the way to color the network
            choose one of ['flat', 'fiber', 'fiber_length', 'fiber_volume', 'segment',
            'segment_length', 'segment_volume', 'fiber_segment_ratio']
        :param str color_map: name of a matplotlib colormap
        :param tuple(float) color: color if color_mode is `'flat'`
        :param str object_type: when set to `'line'` render center line
        :param tuple segment_list: limit the visualisation to certain segment (use sid)
        """

        if self.network.extractor_steps is None:
            self.logger.warn('Not Available! The network was not initialized with an image.')
            return
        elif not self.network.extractor_steps:
            self.logger.warn('Not Available! The extraction steps were not saved (see config).')
            return

        original_spacing = [self.network.extractor_data['xy_spacing']] * 2 + [self.network.extractor_data['z_spacing']]
        cubic_spacing = [self.network.extractor_data['xy_spacing']] * 3

        original = self.network.extractor_steps['is_raw']
        original_vol = vedo.Volume(original, spacing=original_spacing)
        original_threshold = filters.threshold_otsu(original)

        z_drop = self.network.extractor_steps['is_z_drop']
        z_drop_vol = vedo.Volume(z_drop, spacing=original_spacing)
        z_drop_threshold = filters.threshold_otsu(z_drop)

        binary_vol = vedo.Volume(self.network.extractor_steps['is_final'], spacing=cubic_spacing)

        object_list = self.__set_up_objects(color_mode, color, color_map, object_type, segment_list)

        text1 = vedo.Text2D('Original image', c='black')
        text2 = vedo.Text2D('Z-drop image', c='black')
        text3 = vedo.Text2D('Binary image', c='black')
        text4 = vedo.Text2D(f'Reconstruction', c='black')

        vedo.show(((original_vol.isosurface(original_threshold), text1),
                   (z_drop_vol.isosurface(z_drop_threshold), text2),
                   (binary_vol.isosurface(1), text3),
                   object_list + [text4]), N=4)
        vedo.plotter.closePlotter()

    @property
    def raster(self):
        if 'raster' in self.storage:
            return self.storage['raster']
        else:
            # draw the network in a ~1E9 voxel sized space
            self.raster_resolution = float(np.cbrt(1E8 / self.network.bbox_volume))
            self.logger.info(f'Rasterizing network (voxel resolution : {self.raster_resolution:.2E})')
            result, offset = self._rasterize_network(self.network, resolution=self.raster_resolution, debug=self.logger.level <= 10)
            self.storage['raster'] = result
            self.raster_offset = offset
            return result

    @classmethod
    def _rasterize_network(cls, net, resolution=1.0, segment_list=None, debug=False):
        max_r = net.max_radius
        dimension = np.array((net.bbox_size + np.ceil(max_r * 3)) * resolution, dtype=int)
        offset = np.array((net.bbox[0] - np.ceil(max_r*1.5)) * resolution, dtype=int)
        exact_offset = (net.bbox[0] - np.ceil(max_r*1.5)) * resolution
        base = np.zeros(dimension, dtype=bool)

        work_list = []
        if segment_list is None:
            segment_list = net.segment.keys()
        for sid in segment_list:
            segment = net.segment[sid]
            for n in range(len(segment.point) - 1):
                start_point = segment.point[n] * resolution - offset
                stop_point = segment.point[n + 1] * resolution - offset
                start_radius = segment.radius[n] * resolution
                stop_radius = segment.radius[n + 1] * resolution
                work_list.append((start_point, stop_point, start_radius, stop_radius))
        if config.core_count < 1:
            core_count = cpu_count()
        else:
            core_count = config.core_count
        chunk_size = 20
        work_size = len(work_list)
        chunk_number = work_size // chunk_size
        work_list = list((work_list[cn*chunk_size:(cn+1)*chunk_size] for cn in range(chunk_number+1)))

        with Pool(processes=core_count) as pool:
            if debug:
                with tqdm(total=work_size) as pbar:
                    for result in pool.imap_unordered(partial(cls._raster_worker, dimension=dimension), work_list,
                                                      chunksize=1):
                        for included_points in result:
                            base[included_points[:, 0], included_points[:, 1], included_points[:, 2]] = 1
                            pbar.update()
            else:
                for result in pool.imap_unordered(partial(cls._raster_worker, dimension=dimension), work_list, chunksize=1):
                    for included_points in result:
                        base[included_points[:, 0], included_points[:, 1], included_points[:, 2]] = 1
        base = ndimage.binary_closing(base, iterations=5)
        return np.transpose(base, (2, 1, 0)), np.array((offset[2], -offset[1], offset[0]))

    @staticmethod
    def _raster_worker(work_list, dimension=None):
        def points_in_conical_frustum(start_point, stop_point, max_radius, radius_lambda):
            vector = stop_point - start_point
            if np.all(vector == 0):
                return np.ceil(start_point)[:, np.newaxis]
            box_start = np.min(np.stack((start_point, stop_point)), axis=0) - [np.ceil(max_radius)] * 3
            box_start[box_start < 0] = 0
            box_start = (np.floor(box_start).astype(np.uint16))
            box_stop = np.max(np.stack((start_point, stop_point)), axis=0) + [np.ceil(max_radius)] * 3
            box_stop[box_stop > dimension] = dimension[box_stop > dimension]
            box_stop = (np.ceil(box_stop).astype(np.uint16))
            box_shape = box_stop - box_start
            box_shape[box_shape < 1] = 1
            raster_points = np.array(list(np.ndindex(*box_shape))) + box_start
            vector_length = np.linalg.norm(vector)
            partial_occupation = np.dot((raster_points - start_point), vector) / (vector_length ** 2)
            distance = np.linalg.norm(np.cross(vector, start_point - raster_points[(0 <= partial_occupation) & (partial_occupation <= 1)]),
                                      axis=1) / vector_length
            if radius_lambda is None:
                return raster_points[(0 <= partial_occupation) & (partial_occupation <= 1)][distance <= max_radius]
            else:
                return raster_points[(0 <= partial_occupation) & (partial_occupation <= 1)][
                    distance <= radius_lambda(partial_occupation[(0 <= partial_occupation) & (partial_occupation <= 1)])]

        result = []
        for task in work_list:
            # task == (start_point, stop_point, start_radius, stop_radius)
            if task[2] == task[3]:
                included_points = points_in_conical_frustum(task[0], task[1], task[2], None)
            else:
                included_points = points_in_conical_frustum(task[0], task[1], np.max((task[2], task[3])),
                                                            lambda p: task[2] + (task[3] - task[2]) * p)
            if included_points is not None:
                result.append(included_points)
        return result

    @staticmethod
    def show_3d_image(image, name, binary=False, spacing=None):
        """
        Visualize a image stack interactively. The threshold can be altered in the opened window.

        :param np.ndarray image: image stack
        :param str name: display name
        :param bool binary: if the image stack is already binary
        :param tuple(float) spacing: spacing in all three axis
        """
        vp = vedo.Plotter(axes=1)
        vedo.settings.useParallelProjection = True
        vol = vedo.Volume(image, spacing=spacing)
        vol_text = vedo.Text2D(name, c='black')

        if binary:
            window = vp.show((vol.isosurface(1), vol_text))
        else:
            threshold = filters.threshold_otsu(image)
            iso_surface = vol.isosurface(threshold)
            image_max = np.max(image)

            def iso_slider(widget, event):
                previous_actor = vp.actors[0]
                slider_value = widget.GetRepresentation().GetValue()
                real_value = image_max * slider_value / 100
                new_actor = vol.isosurface(real_value).alpha(iso_surface.alpha())
                vp.renderer.RemoveActor(previous_actor)
                vp.renderer.AddActor(new_actor)
                vp.actors = [new_actor] + vp.actors[1:]

            vp.addSlider2D(iso_slider, xmin=0, xmax=100, value=(threshold / image_max) * 100, pos=4,
                           title='threshold', showValue=True, )
            window = vp.show((iso_surface, vol_text))
        window.close()
        #vedo.closePlotter()

    @classmethod
    def save_3d_image(cls, image, out_path='.', overwrite=False, image_resolution=None, binary=False,
                      threshold=None, spacing=None,  color=None, background=None,
                      azimuth=None, elevation=None, roll=None, rgba=None):
        """
        Render thresholded image stack to file.

        :param np.ndarray image: image stack
        :param out_path: file or folder path where to save the network, if `None` show the plot.
        :type out_path: str, Path
        :param bool overwrite: allow file overwrite
        :param int image_resolution: image width
        :param bool binary: if the image stack is already binary
        :param float threshold: threshold for volume creation in percent
        :param tuple(float) spacing: spacing in all three axis
        :param tuple(float) color: color for rendering (0.0-1.0)
        :param tuple(float) background: background color
        :param float azimuth: change camera azimuth
        :param float elevation: change camera elevation
        :param float roll: roll camera
        :param bool rgba: allow transparency in saved file
        :return: path to saved file
        :rtype: Path
        """

        out_path, needs_unlink = helper.out_path_check(out_path, prefix='threshold_3D', suffix='.png', overwrite=overwrite)

        if out_path is None:
            return

        if image_resolution is None:
            image_resolution = config.render.image_resolution
        if color is None:
            color = config.render.color
        if background is None:
            background = config.render.background
        if rgba is None:
            rgba = config.render.rgba

        helper.notebook_render_backend()
        vedo.settings.useParallelProjection = False
        vol = vedo.Volume(image, spacing=spacing)
        if binary:
            vol = vol.isosurface(1)
        else:
            if threshold is None:
                threshold = filters.threshold_otsu(image)
            else:
                threshold = np.max(image) * threshold / 100
            vol = vol.isosurface(threshold)

        vol.color(color)
        vp = vedo.Plotter(axes=0, offscreen=True, interactive=False, size=(image_resolution, image_resolution),
                          bg=background)
        vedo_obj = vp.show([vol], azimuth=azimuth, roll=roll, elevation=elevation)

        if needs_unlink:
            out_path.unlink()
        cls._base_render(image_name=out_path.name, work_dir=out_path.parent, cut_to_fit=True, rgba=rgba)
        vedo_obj.close()
        if config.render.notebook:
            helper.notebook_display_backend()
        return out_path

    def export_x3d(self, out_path='.', overwrite=False,
                   color_mode='flat', color_map='jet', color=None, object_type=None, segment_list=None,
                   azimuth=None, elevation=None, roll=None):
        """

        :param out_path: file or folder path where to save the network, if `None` show the plot.
        :type out_path: str, Path
        :param bool overwrite: allow file overwrite
        :param str color_mode: sets the way to color the network
            choose one of ['flat', 'fiber', 'fiber_length', 'fiber_volume', 'segment',
            'segment_length', 'segment_volume', 'fiber_segment_ratio']
        :param str color_map: name of a matplotlib colormap
        :param tuple(float) color: color if color_mode is `'flat'`
        :param str object_type: when set to `'line'` render center line
        :param tuple segment_list: limit the visualisation to certain segment (use sid)
        :param float azimuth: change camera azimuth
        :param float elevation: change camera elevation
        :param float roll: roll camera
        :return: path to saved file
        :rtype: Path
        """

        out_path, needs_unlink = helper.out_path_check(out_path, network=self.network, prefix='', suffix='.static',
                                                       overwrite=overwrite, logger=self.logger)
        if out_path is None:
            return

        object_list = self.__set_up_objects(color_mode, color, color_map, object_type, segment_list)
        vedo.settings.useParallelProjection = True
        vedo_obj = vedo.show(object_list, axes=0, offscreen=True, azimuth=azimuth, roll=roll, elevation=elevation)

        exporter = vtk.vtkX3DExporter()
        exporter.SetBinary(False)
        exporter.FastestOff()
        exporter.SetInput(vedo.settings.plotter_instance.window)
        exporter.SetFileName(str(out_path.absolute()))
        exporter.Update()
        exporter.Write()
        vedo_obj.close()
        return out_path

    def export_image_stack(self, out_path='.', overwrite=False,
                           voxel_resolution=None,  segment_list=None):
        """
        Export a network as TIFF image stack.

        :param out_path: file or folder path where to save the network, if `None` show the plot.
        :type out_path: str, Path
        :param bool overwrite: allow file overwrite
        :param float voxel_resolution: number of voxels per unit length
        :param tuple segment_list: limit the visualisation to certain segment (use sid)
        :return: path to saved file
        :rtype: Path
        """
        prefix = f'image_stack_'
        out_path, needs_unlink = helper.out_path_check(out_path, network=self.network, prefix=prefix, suffix='.tif',
                                                       overwrite=overwrite, logger=self.logger)
        if out_path is None:
            return

        if voxel_resolution is None:
            image = self.raster
            voxel_resolution = self.raster_resolution
        else:
            self.logger.info(f'Rasterizing network (voxel resolution : {voxel_resolution:.2E} voxel/unit)')
            image, offset = self._rasterize_network(self.network, resolution=voxel_resolution, segment_list=segment_list)
        image = ndimage.gaussian_filter(image.astype(np.float32), 1.0)
        tif.imwrite(str(out_path.absolute()), (image * 255).astype(np.uint8), bigtiff=True, compression='DEFLATE',
                    resolution=(voxel_resolution, voxel_resolution, None),
                    metadata={'spacing': voxel_resolution, 'unit': 'um', 'axes': 'ZYX'})
        return out_path

    def export_intensity_projection(self, image=None, out_path='.', overwrite=False, color_map='bone',
                                    mode='max', axis='z', voxel_resolution=None, segment_list=None):
        """
        Create a intensity projection of the rasterized network.

        :param out_path: file or folder path where to save the network, if `None` show the plot.
        :type out_path: str, Path
        :param bool overwrite: allow file overwrite
        :param str mode: projection mode, choose between `'max'` or `'average'`
        :param str axis: project along this axis (x, y, z)
        :param float voxel_resolution: number of voxels per unit length
        :param tuple segment_list: limit the visualisation to certain segment (use sid)
        :return: path to saved file
        :rtype: Path
        """

        prefix = f'max_intensity_{axis}_'
        out_path, needs_unlink = helper.out_path_check(out_path, network=self.network, prefix=prefix, suffix='.png',
                                                       overwrite=overwrite, logger=self.logger)
        if out_path is None:
            return
        if image is None:
            raster = True
            if voxel_resolution is None:
                image = self.raster
                voxel_resolution = self.raster_resolution
            else:
                self.logger.info(f'Rasterizing network (voxel resolution : {voxel_resolution:.2E} voxel/unit)')
                image = self._rasterize_network(self.network, resolution=voxel_resolution, segment_list=segment_list)
        else:
            raster = False
        if axis not in (0, 1, 2):
            if axis == 'x':
                axis = 0
            elif axis == 'y':
                axis = 1
            elif axis == 'z':
                axis = 2
            else:
                axis = 0
        if mode == 'max':
            image = np.max(image, axis=axis)
        elif mode == 'average':
            image = np.average(image, axis=axis)
        else:
            self.logger.warning(f'Mode {mode} not supported.')
            return None
        # image = np.transpose(image, (1, 0))
        image = image.astype(float)
        cmp = cm.get_cmap(color_map)
        image = (image - np.min(image)) / (np.max(image) - np.min(image))
        image = cmp(image)
        image = (image*255).astype(np.uint8)
        Image.fromarray(image, "RGBA").save(out_path.absolute())

        #     return image_name
        #     tif.imsave(str(out_path.absolute()), (image * 255).astype(np.uint8), compression='DEFLATE',
        #                resolution=(voxel_resolution, voxel_resolution, None),
        #                metadata={'unit': 'um', 'axes': 'YX'})
        # else:
        #     tif.imsave(str(out_path.absolute()), (image * 255).astype(np.uint8), compression='DEFLATE')

        return out_path

    def overview(self, out_path='.', overwrite=False, image_resolution=None,
                 color_mode='flat', color_map='jet', color=None, background=None, object_type=None, segment_list=None,
                 azimuth=None, elevation=None, roll=None, rgba=None, axes=0):
        """

        :param out_path: file or folder path where to save the network, if `None` show the plot.
        :type out_path: str, Path
        :param bool overwrite: allow file overwrite
        :param int image_resolution: image width
        :param str color_mode: sets the way to color the network
            choose one of ['flat', 'fiber', 'fiber_length', 'fiber_volume', 'segment',
            'segment_length', 'segment_volume', 'fiber_segment_ratio']
        :param str color_map: name of a matplotlib colormap
        :param tuple(float) color: color if color_mode is `'flat'`
        :param tuple(float) background: background color
        :param str object_type: when set to `'line'` render center line
        :param tuple segment_list: limit the visualisation to certain segment (use sid)
        :param float azimuth: change camera azimuth
        :param float elevation: change camera elevation
        :param float roll: roll camera
        :param bool rgba: allow transparency in saved file
        :param axes: vedo axis selection (`Documentation <https://vedo.embl.es/autodocs/content/vedo/plotter.html#show>`_)
        :return: path to saved file
        :rtype: Path
        """

        prefix = f'overview_{color_mode}_'
        out_path, needs_unlink = helper.out_path_check(out_path, network=self.network, prefix=prefix, suffix='.png',
                                                       overwrite=overwrite, logger=self.logger)
        if out_path is None:
            return

        if image_resolution is None:
            image_resolution = config.render.image_resolution

        if background is None:
            background = config.render.background

        object_list = self.__set_up_objects(color_mode, color, color_map, object_type, segment_list)
        vedo.settings.useParallelProjection = True
        helper.notebook_render_backend()

        vp = vedo.Plotter(axes=axes, offscreen=True, interactive=False, size=(image_resolution, image_resolution),
                          bg=background)

        vedo_obj = vp.show(object_list, azimuth=azimuth, roll=roll, elevation=elevation)
        if rgba is None:
            rgba = config.render.rgba

        if needs_unlink:
            out_path.unlink()
        self._base_render(image_name=out_path.name, work_dir=out_path.parent, cut_to_fit=True, rgba=rgba)

        vedo_obj.close()

        self.logger.info(f"New overview saved under: {out_path.absolute()}")
        if config.render.notebook:
            helper.notebook_display_backend()

        return out_path

    def animation(self, out_path='.',  overwrite=False, duration=3, fps=30, height=None,
                  color_mode='fiber',  color_map='jet', color=None, background=None,
                  object_type=None, segment_list=None, rgba=False, zoom=None):
        """
        Animate a network by rotate the camera around it. Saves as h264 :file:`.mp4` file by default.
        Supports also :file:`.webm` and :file:`.gif` as target.

        :param out_path: file or folder path where to save the network, if `None` show the plot.
        :type out_path: str, Path
        :param bool overwrite: allow file overwrite
        :param float duration: animation duration in seconds
        :param int fps: frames per second
        :param int height: height of the animation (16:9 format)
        :param str color_mode: sets the way to color the network
            choose one of ['flat', 'fiber', 'fiber_length', 'fiber_volume', 'segment',
            'segment_length', 'segment_volume', 'fiber_segment_ratio']
        :param str color_map: name of a matplotlib colormap
        :param tuple(float) color: color if color_mode is `'flat'`
        :param tuple(float) background: background color
        :param str object_type: when set to `'line'` render center line
        :param tuple segment_list: limit the visualisation to certain segment (use sid)
        :param bool rgba: allow transparency in saved file (for ``.gif`` and ``.webm``)
        :param float zoom: zoom by rendering a larger image and cutting it down afterwards (must be > 1.0)
        :return: path to saved file
        :rtype: Path
        """

        self.logger.info('Preparing animation')
        prefix = f'animation_{color_mode}_'
        out_path, needs_unlink = helper.out_path_check(out_path, network=self.network, prefix=prefix, suffix='.mp4',
                                                       overwrite=overwrite, logger=self.logger)
        if out_path is None:
            return

        if out_path.suffix not in ['.gif', '.webm']:
            rgba = False

        if rgba:
            video_color = 'yuva420p'
        else:
            video_color = 'yuv420p'

        object_list = self.__set_up_objects(color_mode, color, color_map, object_type, segment_list)

        if height is None:
            height = config.render.animation_height
        if background is None:
            background = config.render.background

        helper.notebook_render_backend()
        vedo.settings.useParallelProjection = False

        width = int(height / 9 * 16)
        height = int(round(height/2))*2
        width = int(round(width/2))*2

        if zoom is not None:
            if isinstance(zoom, (float, int)):
                if zoom < 1.0:
                    raise ValueError
                new_height = int(((height * zoom) // 2) * 2)
                new_width = int(new_height / 9 * 16)

                offset_ver = int((new_height - height)//2)
                offset_hor = int((new_width - width)//2)
            else:
                if zoom[0] < 1.0 or zoom[1] < 1.0:
                    raise ValueError
                new_height = int(((height * zoom[1]) // 2) * 2)
                new_width = int(new_height / 9 * 16)
                offset_ver = int((new_height - height) // 2)
                offset_hor = int(((new_width/zoom[0]//2)*2) // 2)
            # top: bottom, left: right
            cut = (offset_ver, -offset_ver, offset_hor, -offset_hor)
            width = new_width
            height = new_height
        else:
            cut = None

        vp = vedo.Plotter(axes=0, offscreen=True, interactive=False, size=(width, height), bg=background)
        vp.show(object_list)

        work_base = ''.join(random.choices(string.ascii_lowercase, k=8))
        with TemporaryDirectory(prefix=self.work_dir_prefix) as tmp_dir:
            tmp_dir_path = Path(tmp_dir)
            work_dir = tmp_dir_path / work_base
            work_dir.mkdir(parents=True)
            frames = int(duration * fps)

            self.logger.debug(f'Generating {frames} images under {work_dir.absolute()}')
            if config.log_level <= 20:
                work_list = tqdm(range(frames), desc='rendering', unit='frame', total=frames)
            else:
                work_list = range(frames)

            for n in work_list:
                vp.show(azimuth=(360 / frames))
                self._base_render(image_name=f'ANI_{n:07}.png', work_dir=work_dir, rgba=rgba, cut=cut)

            if out_path.suffix == '.gif':
                command = [config.render.ffmpeg_path, '-r', str(fps), '-f', "image2", "-i", "ANI_%07d.png",
                           '-filter_complex', f'[0:v] fps={fps}, split [a][b];[a] palettegen [p];[b][p] paletteuse',
                           '_Animation.gif']
                self.logger.debug(f'Combine images ({" ".join(command)})')
                subprocess.call(command, cwd=work_dir, stdout=self.terminal_out, stderr=self.terminal_out)
            elif out_path.suffix == '.webm':
                command = [config.render.ffmpeg_path, '-r', str(fps), '-f', "image2", "-i", "ANI_%07d.png",
                           '-c:v', 'libvpx-vp9', '-b:v', '0', '-crf', '20', "-pix_fmt", video_color, '-pass', '1', '-an', '-f', 'null',
                           os.devnull]
                self.logger.debug(f'Combine images first pass ({" ".join(command)})')
                subprocess.call(command, cwd=work_dir, stdout=self.terminal_out, stderr=self.terminal_out)
                command = [config.render.ffmpeg_path, '-r', str(fps), '-f', "image2", "-i", "ANI_%07d.png",
                           '-c:v', 'libvpx-vp9', '-b:v', '0', '-crf', '20', "-pix_fmt", video_color, '-pass', '2', '-an', "_Animation.webm"]
                self.logger.debug(f'Combine images second pass ({" ".join(command)})')
                subprocess.call(command, cwd=work_dir, stdout=self.terminal_out, stderr=self.terminal_out)
            else:
                command = [config.render.ffmpeg_path, '-r', str(fps), '-f', "image2", "-i", "ANI_%07d.png",
                           "-vcodec", "libx264", "-crf", "20", "-pix_fmt", video_color, "_Animation.mp4"]
                self.logger.debug(f'Combine images ({" ".join(command)})')
                subprocess.call(command, cwd=work_dir, stdout=self.terminal_out, stderr=self.terminal_out)

            in_path = work_dir / f"_Animation"
            if needs_unlink:
                out_path.unlink()
            shutil.move(str(in_path.with_suffix(out_path.suffix).absolute()), str(out_path.absolute()))
            self.logger.info(f"New animation saved under: {out_path.absolute()}")
            if config.render.notebook:
                helper.notebook_display_backend()
        return out_path

    @classmethod
    def _base_render(cls, image_name=None, work_dir=None, cut_to_fit=False, rgba=False, cut=None):

        if work_dir is None:
            cls.logger.warn('No work folder available to render data into.')

        w2if = vtk.vtkWindowToImageFilter()
        w2if.SetInput(vedo.settings.plotter_instance.window)
        w2if.SetInputBufferTypeToRGBA()
        w2if.ReadFrontBufferOff()  # read from the back buffer
        w2if.Update()

        w2if_out = w2if.GetOutput()
        image_raw = vtk_to_numpy(w2if_out.GetPointData().GetArray("ImageScalars"))
        image_raw = image_raw[:, [0, 1, 2, 3]]

        ydim, xdim, _ = w2if_out.GetDimensions()
        image_raw = image_raw.reshape([xdim, ydim, -1])
        image_data = np.flip(image_raw, axis=0)
        if cut_to_fit:
            Y = np.average(image_data[:, :, 3], axis=0) != 0
            X = np.average(image_data[:, :, 3], axis=1) != 0
            top, bottom = np.argmax(X), len(X) - np.argmax(np.flip(X))
            left, right = np.argmax(Y), len(Y) - np.argmax(np.flip(Y))
            image_data = image_data[top:bottom, left:right, :]
        if cut is not None:
            image_data = image_data[cut[0]:cut[1], cut[2]:cut[3], :]

        if image_name is None:
            image_name = ''.join(random.choices(string.ascii_lowercase + string.digits, k=20)) + '.png'

        if rgba:
            Image.fromarray(image_data, "RGBA").save(work_dir / image_name)
        else:
            Image.fromarray(image_data[:, :, :3], "RGB").save(work_dir / image_name)

        return image_name
