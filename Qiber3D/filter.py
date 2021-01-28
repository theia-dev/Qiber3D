import networkx as nx
import numpy as np

import Qiber3D


class Filter:
    @staticmethod
    def length(net, length):
        """
        :param net: Original Network
        :type net: Qiber3D.Network
        :param length:
        :return: Filtered network
        :rtype: Qiber3D.Network
        """
        return Filter.fiber_attribute(net, 'length', length)

    @staticmethod
    def volume(net, volume):
        """
        :param net: Original Network
        :type net: Qiber3D.Network
        :param volume:
        :return: Filtered network
        :rtype: Qiber3D.Network
        """
        return Filter.fiber_attribute(net, 'volume', volume)

    @staticmethod
    def fiber_attribute(net, attribute, value):
        filtered_segments = []
        if hasattr(value, '__len__'):
            if len(value) == 1:
                for fiber in net.fiber.values():
                    if getattr(fiber, attribute) >= value[0]:
                        filtered_segments.extend(fiber.segment)
            elif len(value) == 2:
                for fiber in net.fiber.values():
                    if value[0] <= getattr(fiber, attribute) <= value[1]:
                        filtered_segments.extend(fiber.segment)
            else:
                return None
        else:
            for fiber in net.fiber.values():
                if getattr(fiber, attribute) >= value:
                    filtered_segments.extend(fiber.segment)

        return Qiber3D.IO.load.network(net, segment_list=filtered_segments)

    @staticmethod
    def segment_attribute(net, attribute, value):
        filtered_segments = []
        if hasattr(value, '__len__'):
            if len(value) == 1:
                for segment in net.segment.values():
                    if getattr(segment, attribute) >= value[0]:
                        filtered_segments.append(segment.sid)
            elif len(value) == 2:
                for segment in net.segment.values():
                    if value[0] <= getattr(segment, attribute) <= value[1]:
                        filtered_segments.append(segment.sid)
            else:
                return None
        else:
            for segment in net.segment.values():
                if getattr(segment, attribute) >= value:
                    filtered_segments.append(segment.sid)

        return Qiber3D.IO.load.network(net, segment_list=filtered_segments)

    @staticmethod
    def loop(net, attribute='length', mode='min'):
        filtered_segments = []
        if mode == 'min':
            weight = attribute
        elif mode == 'max':
            weight = f'tree_max_{attribute}'
        else:
            return None

        for fiber in net.fiber.values():
            graph_tree = nx.minimum_spanning_tree(fiber.graph, weight=weight)
            filtered_segments.extend(list(nx.get_edge_attributes(graph_tree, 'sid').values()))

        return Qiber3D.IO.load.network(net, segment_list=filtered_segments)

    @staticmethod
    def volume_ratio(net, ratio=0.5, min_volume=None):
        """
        :param net: Original Network
        :type net: Qiber3D.Network
        """
        filtered_segments = []
        resolution = float(np.cbrt(1E8 / net.bbox_volume))
        for fiber in net.fiber.values():
            raster_volume = np.sum(Qiber3D.Render._rasterize_network(net, segment_list=fiber.segment,
                                                                     resolution=resolution)) / (resolution**3)
            print(raster_volume/fiber.volume, raster_volume, fiber.volume)

        return Qiber3D.IO.load.network(net, segment_list=filtered_segments)

    @staticmethod
    def branch(net, has_branch=True):
        filtered_segments = []
        for fiber in net.fiber.values():
            if len(fiber.segment) > 1 and has_branch:
                filtered_segments.extend(fiber.segment)
            elif len(fiber.segment) == 1 and not has_branch:
                filtered_segments.extend(fiber.segment)

        return Qiber3D.IO.load.network(net, segment_list=filtered_segments)