from itertools import combinations

import networkx as nx
import numpy as np
from scipy import interpolate
from scipy import ndimage
from skimage.morphology import skeletonize_3d
from tqdm import tqdm

import Qiber3D


class Reconstruct:  # BJH

    logger = Qiber3D.helper.get_logger()

    @staticmethod
    def __paths_to_edges(paths):
        edges = []
        for path in paths:
            edges += [(path[idx], path[idx + 1]) for idx in range(len(path) - 1)]
        return edges

    @classmethod
    def __path_length(cls, graph, path):
        return sum((graph.get_edge_data(*edge)['length'] for edge in cls.__paths_to_edges((path,))))

    @staticmethod
    def __combine_radius_points(new_points, add_points, new_radius, add_radius):
        if new_points[0] in (add_points[0], add_points[-1]):
            if new_points[0] == add_points[0]:
                add_points.reverse()
                add_radius.reverse()
            new_points = add_points[:-1] + new_points
            new_radius = add_radius[:-1] + new_radius
        elif new_points[-1] in (add_points[0], add_points[-1]):
            if new_points[-1] == add_points[-1]:
                add_points.reverse()
                add_radius.reverse()
            new_points = new_points + add_points[1:]
            new_radius = new_radius + add_radius[1:]
        return new_points, new_radius

    @classmethod
    def __extend_segment_by_path(cls, net, sid, start_node, graph, path):
        segment = net.segment[sid]
        cls.logger.debug(f'Extending {sid} by path {path})')
        new_points = segment.point.tolist()
        new_radius = segment.radius.tolist()
        if start_node == path[0]:
            pass
        elif start_node == path[-1]:
            path.reverse()
        else:
            raise ValueError(f'Start node {start_node} is not on the edge of the path [{path[0]}] <-> [{path[-1]}]')
        for edge in cls.__paths_to_edges((path,)):
            edge_data = graph.get_edge_data(*edge)
            add_points = net.segment[edge_data['sid']].point.tolist()
            add_radius = net.segment[edge_data['sid']].radius.tolist()
            new_points, new_radius = cls.__combine_radius_points(new_points, add_points, new_radius, add_radius)
        segment.point = np.array(new_points)
        segment.radius = np.array(new_radius)
        net.segment[sid] = segment

    @classmethod
    def __add_segment_from_path(cls, net, graph, path):
        cls.logger.debug(f'Create new segment from path {path}')
        edges = cls.__paths_to_edges((path,))
        start_data = graph.get_edge_data(*edges[0])
        new_points = net.segment[start_data['sid']].point.tolist()
        new_radius = net.segment[start_data['sid']].radius.tolist()
        if len(edges) > 1:
            for edge in edges[1:]:
                edge_data = graph.get_edge_data(*edge)
                add_points = net.segment[edge_data['sid']].point.tolist()
                add_radius = net.segment[edge_data['sid']].radius.tolist()
                new_points, new_radius = cls.__combine_radius_points(new_points, add_points, new_radius, add_radius)
        new_id = max(list(net.segment)) + 1
        net.segment[new_id] = Qiber3D.Segment(radius=np.array(new_radius), point=np.array(new_points),
                                              segment_index=new_id)

    @classmethod
    def __join_segments(cls, net, segment_ids):
        cls.logger.debug(f'Join segments {segment_ids}')
        new_id = max(list(net.segment)) + 1
        new_points = net.segment[segment_ids[0]].point.tolist()
        new_radius = net.segment[segment_ids[0]].radius.tolist()
        add_points = net.segment[segment_ids[1]].point.tolist()
        add_radius = net.segment[segment_ids[1]].radius.tolist()
        new_points, new_radius = cls.__combine_radius_points(new_points, add_points, new_radius, add_radius)
        net.segment[new_id] = Qiber3D.Segment(radius=np.array(new_radius), point=np.array(new_points), segment_index=new_id)

    @staticmethod
    def __join_branch_point(net, point, segment_ids):
        new_points = []
        new_radius = []
        segment_todo = []
        for sid in segment_ids:
            start_dist = np.sum((net.segment[sid].point[0] - point) ** 2)
            end_dist = np.sum((net.segment[sid].point[-1] - point) ** 2)
            idx = -int(start_dist > end_dist)
            segment_todo.append((sid, idx))
            new_points.append(net.segment[sid].point[idx])
            new_radius.append(net.segment[sid].radius[idx])

        if len(new_points) > 1:
            new_point = np.average(new_points, axis=0)
            new_radius = np.average(new_radius)
            for sid, idx in segment_todo:
                net.segment[sid].point[idx] = new_point
                net.segment[sid].radius[idx] = new_radius

    @classmethod
    def clean(cls, net, sliver_threshold=6):
        """
        Clean a :class:`Qiber3D.Network` from small sliver.

        :param Qiber.Network net: network to smooth
        :param int sliver_threshold: treat smaller segments
        :return:
        """
        cls.logger.info(f'Cleaning Network')
        to_delete = []
        sliver_threshold = max(6, sliver_threshold)
        for fiber in net.fiber.values():
            invalid_edges = []
            for f, t, data in fiber.graph.edges(data=True):
                if data['length'] < sliver_threshold:
                    invalid_edges.append((f, t))
                    to_delete.append(data['sid'])

            invalid_graph = fiber.graph.edge_subgraph(invalid_edges).copy()
            for conglomerate in (invalid_graph.subgraph(c).copy() for c in nx.connected_components(invalid_graph)):
                outer_nodes = {}
                for node in conglomerate.nodes:
                    out_length = 0
                    connections = []
                    icc = 0
                    for f, t, data in fiber.graph.edges(node, data=True):
                        if data['sid'] not in to_delete:
                            out_length += data['length']
                            connections.append((data['sid'], data['length']))
                        else:
                            icc += 1
                    if connections:
                        outer_nodes[node] = {'length': out_length, 'cc': len(connections), 'icc': icc,
                                             'connections': connections, 'node': node}

                # isolated island of to short segments
                if len(outer_nodes) == 0:
                    cls.logger.debug(f'Treat isolated conglomerate with {len(conglomerate.edges)} edges')
                    # check if a path is possible through the island that is over the dust_threshold
                    possible_paths = [path for path in
                                      sum([list(entry.values()) for entry in
                                           nx.shortest_path(conglomerate).values()], [])
                                      if len(path) > 1]
                    longest_path = max(possible_paths, key=lambda p: cls.__path_length(conglomerate, p))
                    if cls.__path_length(conglomerate, longest_path) > sliver_threshold:
                        cls.__add_segment_from_path(net, fiber.graph, longest_path)

                # island on the rim of the network
                elif len(outer_nodes) == 1:
                    cls.logger.debug(f'Treat conglomerate at network rim with {len(conglomerate.edges)} edges')
                    node = list(outer_nodes.keys())[0]
                    # add longest 'short' path to the outer segment if just one, else create new segment
                    possible_paths = [entry for entry in nx.shortest_path(conglomerate, node).values() if len(entry) > 1]
                    if possible_paths:
                        longest_path = max(possible_paths, key=lambda p: cls.__path_length(conglomerate, p))
                    else:
                        longest_path = None

                    if outer_nodes[node]['cc'] == 1:
                        cls.__extend_segment_by_path(net, outer_nodes[node]['connections'][0][0], node, fiber.graph,
                                                     longest_path)
                    else:
                        seg_added = False
                        if longest_path is not None:
                            if cls.__path_length(conglomerate, longest_path) > sliver_threshold:
                                cls.__add_segment_from_path(net, fiber.graph, longest_path)
                                seg_added = True

                        if seg_added == False:
                            if outer_nodes[node]['cc'] == 2:
                                sid_list = list((c[0] for c in outer_nodes[node]['connections']))
                                cls.__join_segments(net, sid_list)
                                for sid in sid_list:
                                    to_delete.append(sid)

                # island bridging between two segments
                elif len(outer_nodes) == 2:
                    cls.logger.debug(f'Treat conglomerate separating two segments with {len(conglomerate.edges)} edges')
                    to_connect = sorted(outer_nodes.values(), key=lambda x: x['length'], reverse=True)[:2]
                    to_connect = sorted(to_connect, key=lambda x: x['cc'])

                    # connect via shortest path
                    new_path = nx.shortest_path(conglomerate, to_connect[0]['node'], to_connect[1]['node'], 'length')
                    # extend longer path, or create new segment when two branch point are connected
                    if to_connect[0]['cc'] == 1:
                        cls.__extend_segment_by_path(net, to_connect[0]['connections'][0][0], to_connect[0]['node'],
                                                     conglomerate, new_path)
                    else:
                        cls.__add_segment_from_path(net, fiber.graph, new_path)

                # island bridging between multiple segments
                elif len(outer_nodes) > 2:
                    cls.logger.debug(f'Treat conglomerate connecting {len(outer_nodes)} outer nodes with {len(conglomerate.edges)} edges')

                    outer_nodes_back_up = outer_nodes.copy()
                    try:
                        to_expand = []
                        to_add = []
                        check_graph = nx.Graph()
                        check_graph.add_nodes_from(list(outer_nodes.keys()))
                        while not nx.is_connected(check_graph):
                            new_path_list = []
                            for f, t in combinations(list(outer_nodes.keys()), 2):
                                new_path_list.append(nx.shortest_path(conglomerate, f, t, 'length'))

                            shortest_path = min((path for path in new_path_list), key=lambda p: cls.__path_length(fiber.graph, p))
                            to_connect = sorted((outer_nodes[shortest_path[0]], outer_nodes[shortest_path[-1]]),
                                                key=lambda x: x['length'], reverse=True)
                            to_connect = sorted(to_connect, key=lambda x: x['cc'])
                            check_graph.add_edge(shortest_path[0], shortest_path[-1])
                            if to_connect[0]['cc'] == 1:
                                to_expand.append((net, to_connect[0]['connections'][0][0], to_connect[0]['node'],
                                                           fiber.graph, shortest_path))
                                del outer_nodes[to_connect[0]['node']]
                            else:
                                to_add.append((net, fiber.graph, shortest_path))

                            for edge in cls.__paths_to_edges((shortest_path,)):
                                conglomerate.remove_edge(*edge)

                        for expand_data in to_expand:
                            cls.__extend_segment_by_path(*expand_data)
                        for add_data in to_add:
                            cls.__add_segment_from_path(*add_data)
                    except nx.exception.NetworkXNoPath:
                        rim_points = [net.node_lookup[node] for node in outer_nodes_back_up.keys()]
                        sids = sum([[con[0] for con in node['connections']] for node in outer_nodes_back_up.values()], [])
                        central_point = np.average(rim_points, axis=0)
                        cls.__join_branch_point(net, central_point, sids)

        cls.logger.debug(f'Removing {len(to_delete)} segments.')
        for sid in to_delete:
            del net.segment[sid]

    @classmethod
    def smooth(cls, net, voxel_per_point=10):
        """
        Smooth a :class:`Qiber3D.Network` in place with a third order spline interpolation.

        :param Qiber.Network net: network to smooth
        :param float voxel_per_point: distance between interpolated points
        :return: :class:`Qiber3D.Network`
        """
        cls.logger.info(f'Smooth Segments')
        cpd = net.cross_point_dict
        for sid, segment in net.segment.items():
            resolution = max(5, int(np.floor(segment.length / voxel_per_point)+1))
            base_points = np.zeros((4, len(segment.point),))
            base_points[:3, :] = segment.point.T
            base_points[3, :] = segment.radius
            try:
                tck, u = interpolate.splprep(base_points)
            except TypeError:
                cls.logger.debug(f'Segment {sid} with {len(segment.point)} points was not interpolated')
                continue
            u_fine = np.linspace(0, 1, resolution)
            x, y, z, r = interpolate.splev(u_fine, tck)
            points = np.stack((x, y, z)).T
            segment.point = points
            r[r < 1] = 1
            segment.radius = r

        for point, segment_ids in cpd.items():
            cls.__join_branch_point(net, point, segment_ids)

    @classmethod
    def create_base_network(cls, image, low_memory=False, distance_voxel_overlap=15):
        """
        Create a unoptimized :class:`Qiber3D.Network` from a binary image.

        :param np.ndarray image: binary image stack
        :param bool low_memory: split the image for the euclidean distance transformation
        :param int distance_voxel_overlap: overlap for the low memory euclidean distance transformation in voxel
        :return: :class:`Qiber3D.Network`
        """

        cls.logger.info(f'Skeletonize image by thinning')
        result = skeletonize_3d(image)
        if cls.logger.level <= 10:
            Qiber3D.Render.show_3d_image(result, name='Skeleton', binary=True)

        slice_dict = {
            -1: (np.s_[:-1], np.s_[1:]),
            0: (np.s_[:], np.s_[:]),
            1: (np.s_[1:], np.s_[:-1])
                      }
        search = (
            (0, -1, -1), (-1, 1, -1), (0, -1, 1),
            (0, 1, 0), (-1, 1, 1), (0, 0, -1),
            (1, 0, 1), (1, 1, 0), (1, 0, -1),
            (-1, -1, -1), (-1, -1, 1), (-1, 1, 0),
            (1, 0, 0))

        search_slices = []
        for x, y, z in search:
            search_slices.append(((x, y, z),
                                  list((1 if element == 1 else 0 for element in (x, y, z))),
                                  np.s_[slice_dict[x][0], slice_dict[y][0], slice_dict[z][0]],
                                  np.s_[slice_dict[x][1], slice_dict[y][1], slice_dict[z][1]],
                                  ))

        cls.logger.info(f'Euclidean distance transformation')

        if not low_memory:
            distances = ndimage.distance_transform_edt(image)
        else:
            split = 2
            distances = np.zeros_like(image, dtype=float)
            axs = [[], [], []]
            axr = [[], [], []]
            axd = [[], [], []]
            s = np.array(image.shape) / split
            for n in range(split):
                for ax in range(3):
                    raw_s = (int(max(0, round(n * s[ax] - distance_voxel_overlap))), int(min(round((n + 1) * s[ax] + distance_voxel_overlap), image.shape[ax])))
                    raw_r = (int(max(0, round(n * s[ax]))), int(min(round((n + 1) * s[ax]), image.shape[ax])))
                    raw_d = [raw_r[0]-raw_s[0], raw_r[1]-raw_s[1]]
                    if raw_d[1] == 0:
                        raw_d[1] = image.shape[ax]
                    axs[ax].append(raw_s)
                    axr[ax].append(raw_r)
                    axd[ax].append(raw_d)

            for x in range(split):
                for y in range(split):
                    for z in range(split):
                        distances[axr[0][x][0]:axr[0][x][1], axr[1][y][0]:axr[1][y][1], axr[2][z][0]:axr[2][z][1]] = \
                            ndimage.distance_transform_edt(image[axs[0][x][0]:axs[0][x][1], axs[1][y][0]:axs[1][y][1], axs[2][z][0]:axs[2][z][1]]
                                                           )[axd[0][x][0]:axd[0][x][1], axd[1][y][0]:axd[1][y][1], axd[2][z][0]:axd[2][z][1]]

        graph = nx.Graph()
        volume_shape = image.shape
        if cls.logger.level <= 10:
            search_slices = tqdm(search_slices, desc='Link up skeleton')
        else:
            cls.logger.info(f'Link up skeleton')
            search_slices = search_slices

        for direction, correction, start_slice, end_slice in search_slices:
            from_points = np.argwhere(np.logical_and(result[start_slice], result[end_slice])) + correction
            to_points = from_points - direction
            from_points_ids = np.ravel_multi_index(from_points.T, volume_shape)
            del from_points
            to_points_ids = np.ravel_multi_index(to_points.T, volume_shape)
            del to_points
            graph.add_edges_from(zip(from_points_ids, to_points_ids))

        cls.logger.info(f'Build Qiber3D.Network from the raw graph')
        segment_data = Qiber3D.IO.load._from_graph(graph, point_lookup=volume_shape, radius_lookup=distances,
                                                   ravel=True)
        data = {
            'path': None,
            'name': 'reconstruct',
            'segments': segment_data}
        net = Qiber3D.Network(data)
        return net

    @classmethod
    def get_network(cls, image, scale=1.0, input_path=None, sliver_threshold=6, voxel_per_point=10.0,
                    low_memory=False, distance_voxel_overlap=15):
        """
        Create a cleaned and smoothed :class:`Qiber3D.Network` from a binary image.

        :param np.ndarray image: binary image stack
        :param float scale: ratio between voxel size and length unit
        :param input_path: set this path as new imput path of the returning network
        :type input_path: str, Path
        :param int sliver_threshold: treat smaller segments
        :param float voxel_per_point: distance between interpolated points
        :param bool low_memory: low memory mode
        :param int distance_voxel_overlap: overlap for the low memory euclidean distance transformation in voxel
        :return: :class:`Qiber3D.Network`
        """
        net = cls.create_base_network(image, low_memory=low_memory, distance_voxel_overlap=distance_voxel_overlap)
        cls.clean(net, sliver_threshold=sliver_threshold)
        net = Qiber3D.IO.load.network(net)
        cls.smooth(net, voxel_per_point=voxel_per_point)
        net = Qiber3D.IO.load.network(net, scale=scale, input_path=input_path)
        return net
