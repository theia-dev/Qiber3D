from collections import defaultdict
from pathlib import Path
from textwrap import dedent

import networkx as nx
import numpy as np

from Qiber3D import Render, Figure, IO
from Qiber3D import config, helper


class Segment:
    """
    Class representing the small element in a network

    :param point: ordered list of points forming the Segment
    :type point: ndarray
    :param radius: radii (same order as `point`)
    :type radius: ndarray
    :param segment_index: unique identifier
    :type segment_index: int

    :ivar sid: unique segment identifier
    :vartype aid: int
    :ivar point: ordered points forming the Segment
    :vartype point: ndarray
    :ivar x: ordered list of x coordinates
    :vartype x: ndarray
    :ivar y: ordered list of y coordinates
    :vartype y: ndarray
    :ivar z: ordered list of z coordinates
    :vartype z: ndarray
    :ivar radius: radii in same order as `point` (also available as **r**)
    :vartype radius: ndarray
    :ivar average_radius: average radius
    :vartype average_radius: float
    :ivar cylinder_radius: radius if segment is interpreted as single cylinder
    :vartype cylinder_radius: float
    :ivar diameter: diameters in same order as `point` (also available as **d**)
    :vartype diameter: ndarray
    :ivar average_diameter: average diameters
    :vartype average_diameter: float
    :ivar start: start point coordinates
    :vartype start: tuple
    :ivar end: end point coordinates
    :vartype end: tuple
    :ivar vector: vectors between points
    :vartype vector: ndarray
    :ivar direction: vector pointing from `start` to `end`
    :vartype direction: ndarray
    :ivar length: length from `start` to `end`
    :vartype length: float
    :ivar volume: Segment volume modeled as truncated cones
    :vartype volume: float
    """

    def __init__(self, point, radius, segment_index):
        self.sid = segment_index

        if len(point) < 2:
            raise ValueError

        self.point = point
        self.radius = radius

        self.start = tuple(round(p, 4) for p in self.point[0])
        self.end = tuple(round(p, 4) for p in self.point[-1])
        self.vector = np.diff(self.point, axis=0)
        self.direction = np.sum(self.vector, axis=0)

        self.length = np.sum(np.linalg.norm(self.vector, axis=1))

        self._color = config.render.color
        self.volume = np.sum(np.linalg.norm(self.vector, axis=1) * np.pi / 3 * (
                self.radius[1:] ** 2 + self.radius[:-1] ** 2 + (self.radius[1:]) * (self.radius[:-1])))

    @property
    def r(self):
        return self.radius

    @property
    def diameter(self):
        return self.radius * 2.0

    @property
    def d(self):
        return self.diameter

    @property
    def average_diameter(self):
        return np.average(self.diameter)

    @property
    def average_radius(self):
        return np.average(self.radius)

    @property
    def cylinder_radius(self):
        return np.sqrt((self.volume/self.length)/np.pi)

    @property
    def x(self):
        return self.point[:, 0]

    @property
    def y(self):
        return self.point[:, 1]

    @property
    def z(self):
        return self.point[:, 2]

    def __len__(self):
        return len(self.point)

    def __str__(self):
        info = f"""\
        Segment ID: {self.sid}
          Number of parts: {len(self)}
          Total length: {self.length:.2f}
          Total volume: {self.volume:.2f}
          Average radius: {self.average_radius:.2f}
          Cylinder radius: {self.cylinder_radius:.2f}"""
        return dedent(info)

    def __repr__(self):
        return f'Segment {self.sid} l={self.length:.2f}, V={self.volume:.2f}'


class Fiber:
    """
    Class representing the large elements in a network

    :param network: overarching network
    :type network: :class:`Network`
    :param fiber_id: unique fiber identifier
    :type fiber_id: int
    :param segment_ids: list of segment identifier forming the **Fiber**
    :type segment_ids: list

    :ivar fid: unique fiber identifier
    :vartype aid: int
    :ivar segment: directory of :class:`Segment` forming the :class:`Fiber`
    :vartype aid: dict
    :ivar average_radius: average radius
    :vartype average_radius: float
    :ivar cylinder_radius: radius if segment is interpreted as single cylinder
    :vartype cylinder_radius: float
    :ivar average_diameter: average diameters
    :vartype average_diameter: float
    :ivar length: overall length
    :vartype length: float
    :ivar volume: overall volume modeled as truncated cones
    :vartype volume: float
    :ivar graph: :class:`Fiber` represented as networkx graph
    :vartype graph: nx.Graph
    """

    def __init__(self, network, fiber_id, segment_ids):
        self.fid = fiber_id
        self.sid_list = list(segment_ids)
        self.segment = {sid: network.segment[sid] for sid in segment_ids}
        self.graph = nx.Graph()
        for segment in self.segment.values():
            self.graph.add_edge(network.node_lookup[segment.start], network.node_lookup[segment.end],
                                length=segment.length, radius=segment.cylinder_radius,
                                volume=segment.volume, tree_max_length=-segment.length,
                                tree_max_volume=-segment.volume, sid=segment.sid)


    @property
    def volume(self):
        return sum([segment.volume for segment in self.segment.values()])

    @property
    def length(self):
        return sum([segment.length for segment in self.segment.values()])

    @property
    def cylinder_radius(self):
        return np.sqrt((self.volume / self.length) / np.pi)

    @property
    def average_radius(self):
        return np.average([segment.average_radius for segment in self.segment.values()])

    @property
    def average_diameter(self):
        return 2.0 * np.average([segment.average_radius for segment in self.segment.values()])

    def __len__(self):
        return len(self.sid_list)

    def __str__(self):
        info = f"""\
        Fiber ID: {self.fid}
          Number of segments: {len(self)}
          Total length: {self.length:.2f}
          Total volume: {self.volume:.2f}
          Average radius: {self.average_radius:.2f}
          Cylinder radius: {self.cylinder_radius:.2f}"""
        return dedent(info)

    def __repr__(self):
        return f'Fiber {self.fid} l={self.length:.2f}, V={self.volume:.2f}'


class Network:
    """
    Class representing the complete network

    :param data: metadata and segment data collection
    :type data: dict

    :ivar segment: directory of :class:`Segment` forming the :class:`Network`
    :vartype aid: dict
    :ivar fiber: directory of :class:`Fiber` forming the :class:`Network`
    :vartype aid: dict
    :ivar average_radius: average radius
    :vartype average_radius: float
    :ivar cylinder_radius: radius if segment is interpreted as single cylinder
    :vartype cylinder_radius: float
    :ivar average_diameter: average diameters
    :vartype average_diameter: float
    :ivar length: overall length
    :vartype length: float
    :ivar volume: overall volume modeled as truncated cones
    :vartype volume: float
    :ivar number_of_fibers: fiber count
    :vartype volume: int
    :ivar vector: vectors between points
    :vartype vector: ndarray
    :ivar direction: vector pointing from `start` to `end`
    :vartype direction: ndarray
    :ivar bbox: bounding box corners
    :vartype bbox: ndarray
    :ivar bbox_volume: bounding box volume
    :vartype bbox: float
    :ivar center: bounding box center
    :vartype center: ndarray
    :ivar bbox_size: bounding box size
    :vartype bbox_size: ndarray
    """

    def __init__(self, data):
        self.logger = helper.get_logger()
        if isinstance(data['path'], Path):
            self.input_file = data['path']
            self.input_file_name = self.input_file.name
        else:
            if data['path'] is None:
                self.input_file_name = 'memory'
                self.input_file = None
            else:
                self.input_file_name = Path(data['path'])
                self.input_file = self.input_file.name

        self.name = data['name']
        raw_segments = data['segments']
        self.extractor_steps = None
        self.extractor_data = None
        self.segment = {}
        points = set()
        self.available_segments = list(raw_segments.keys())
        self.available_segments.sort()
        self.cross_point_dict = defaultdict(list)
        for i in self.available_segments:
            i = int(i)
            try:
                self.segment[i] = Segment(raw_segments[i]['points'], raw_segments[i]['radius'], i)
                for point in self.segment[i].point:
                    self.cross_point_dict[(round(point[0], 4), round(point[1], 4), round(point[2], 4))].append(i)
                    points.add((point[0], point[1], point[2]))
            except ValueError:
                self.logger.warning('Missing Segment {i}')
                pass

        self.cross_point_dict = {point: tuple(sids) for point, sids in self.cross_point_dict.items() if len(sids) > 1}
        self.node_lookup = helper.LookUp()

        node_id = 0
        for segment in self.segment.values():
            for key in ('start', 'end'):
                point = getattr(segment, key)
                if point not in self.node_lookup:
                    self.node_lookup[point] = node_id
                    node_id += 1

        self.fiber = self.__cluster_segments()
        self.separate_segments = [fiber.sid_list[0] for fiber in self.fiber.values() if len(fiber) == 1]
        self.clustered_segments = [fiber.sid_list for fiber in self.fiber.values() if len(fiber) > 1]

        self.point = np.array(list(points))
        self.bbox = np.zeros((2, 3), dtype='f8')
        self.bbox[0] = np.min(self.point, axis=0)
        self.bbox[1] = np.max(self.point, axis=0)

        self.center = self.bbox[0] + (self.bbox[1] - self.bbox[0]) / 2.0
        self.bbox_size = self.bbox[1] - self.bbox[0]
        self.bbox_volume = np.product(self.bbox_size)

        self.vector = np.vstack([seg.vector for seg in self.segment.values()])
        self.direction = np.array([seg.direction for seg in self.segment.values()])

        self.spherical_vector = helper.convert_to_spherical(helper.remove_direction(self.vector))
        self.spherical_direction = helper.convert_to_spherical(helper.remove_direction(self.direction))

        self.render = Render(self)
        self.figure = Figure(self)

        pass

    def save(self, out_path='.', overwrite=False, save_steps=False):
        """
        Save network to file.

        :param out_path: file or folder path where to save the network
        :type out_path: str, Path
        :param overwrite: allow file overwrite
        :type overwrite: bool
        :param save_steps: add extraction steps to the saved file
        :type save_steps: bool
        :return: path to saved file
        :rtype: Path
        """
        out_path = IO.export.binary(self, out_path=out_path, overwrite=overwrite, save_steps=save_steps)
        if out_path is not None:
            self.logger.info(f"Network saved to {out_path.absolute()}")
        return out_path

    def export(self, out_path='.', overwrite=False, mode=None, **kwargs):
        """
        Export the network data. Available file types: :file:`.json`, :file:`.qiber`, :file:`.swc`, :file:`.xlsx`,
        :file:`.csv`, :file:`.static`, :file:`.tif` or :file:`.mv3d`. For more details see :class:`Qiber3D.io.IO`.

        :param out_path: file or folder path where to save the network
        :type out_path: str, Path
        :param overwrite:
        :param mode:
        :param kwargs:
        :return:
        """
        out_path = IO.export(self, out_path, overwrite=overwrite, mode=mode, **kwargs)
        if out_path is not None:
            self.logger.info(f"Network exported to {out_path.absolute()}")
        return out_path

    @staticmethod
    def load(path, **kwargs):
        """
        Load a :class:`Network`. Available file types: :file:`.tif`, :file:`.nd2`, :file:`.json`, :file:`.qiber`,
        :file:`.swc`, :file:`.ntr`, :file:`.csv`, or :file:`.mv3d`. For more details see :class:`Qiber3D.io.IO`.

        :param path: file path to input file
        :type path: str, Path
        :return: :class:`Qiber3D.Network`
        """
        return IO.load(path, **kwargs)

    def __cluster_segments(self):
        clusters = []
        sid_connections = set(self.cross_point_dict.values())
        work_list = set(self.available_segments)
        while work_list:
            i = work_list.pop()
            new_cluster = {i}

            connections_to_clean = True
            while connections_to_clean:
                connections_to_clean = []
                for connection in sid_connections:
                    for part in connection:
                        if part in new_cluster:
                            new_cluster.update(connection)
                            connections_to_clean.append(connection)
                            break
                for connection in connections_to_clean:
                    sid_connections.remove(connection)
            clusters.append(new_cluster)
            work_list = work_list.difference(new_cluster)

        fibers = {}
        for fid, cluster in enumerate(clusters):
            fibers[fid] = Fiber(self, fid, cluster)
        return fibers

    @property
    def volume(self):
        return sum([segment.volume for segment in self.segment.values()])

    @property
    def raster_volume(self):
        raw_volume = np.sum(self.render.raster)
        return raw_volume / (self.render.raster_resolution**3)

    @property
    def length(self):
        return sum([segment.length for segment in self.segment.values()])

    @property
    def cylinder_radius(self):
        return np.sqrt((self.volume / self.length) / np.pi)

    @property
    def max_radius(self):
        return np.max([segment.average_radius for segment in self.segment.values()])

    @property
    def average_radius(self):
        return np.average([segment.average_radius for segment in self.segment.values()])

    @property
    def average_diameter(self):
        return 2.0 * np.average([segment.average_radius for segment in self.segment.values()])

    @property
    def number_of_fibers(self):
        return len(self.clustered_segments) + len(self.separate_segments)

    def __str__(self):
        info = f"""\
        Input file: {self.input_file_name}
          Number of fibers: {self.number_of_fibers} (clustered {len(self.clustered_segments)})
          Number of segments: {len(self.segment)}
          Number of branch points: {len(self.cross_point_dict)}
          Total length: {self.length:.2f}
          Total volume: {self.volume:.2f}
          Average radius: {self.average_radius:.3f}
          Cylinder radius: {self.cylinder_radius:.3f}
          Bounding box volume: {self.bbox_volume:.0f}"""
        return dedent(info)

    def __repr__(self):
        return f"Network from '{self.input_file_name}'"
