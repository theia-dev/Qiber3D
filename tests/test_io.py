import unittest
from pathlib import Path
import sys

import numpy as np
from numpy.testing import assert_allclose
import random

sys.path.insert(0, str(Path('.').parent.absolute()))
from Qiber3D import config
from Qiber3D import Network, IO


config.log_level = 40

random.seed('Qiber3D_Testing')


class Model:
    class Load(unittest.TestCase):
        cases = Path(__file__).absolute().resolve().parent / 'cases'
        syn_net = None
        net = None
        places_radius = 3
        places_points = 3
        attribute_list_base = [('length', 1), ('volume', 1), ('average_radius', 2),
                          ('cylinder_radius', 2), ('average_diameter', 2)]
        attribute_list_net = [('number_of_fibers', 0)]

        @classmethod
        def setUpClass(cls):
            cls.syn_net = IO.load.synthetic_network()

        @classmethod
        def tearDownClass(cls):
            del cls.syn_net
            del cls.net

        def test_available_segments(self):
            self.assertEqual(self.syn_net.available_segments, self.net.available_segments)

        def test_radius(self):
            for sid in random.sample(self.net.available_segments, 2):
                for pid in random.sample(range(len(self.net.segment[sid])), 3):
                    self.assertAlmostEqual(self.syn_net.segment[sid].radius[pid],
                                           self.net.segment[sid].radius[pid], places=self.places_radius)

        def test_points(self):
            self.assertEqual(len(self.syn_net.point), len(self.net.point))
            for sid in random.sample(self.net.available_segments, 2):
                for pid in random.sample(range(len(self.net.segment[sid])), 3):
                    self.assertAlmostEqual(self.syn_net.segment[sid].x[pid],
                                           self.net.segment[sid].x[pid], places=self.places_points)
                    self.assertAlmostEqual(self.syn_net.segment[sid].y[pid],
                                           self.net.segment[sid].y[pid], places=self.places_points)
                    self.assertAlmostEqual(self.syn_net.segment[sid].z[pid],
                                           self.net.segment[sid].z[pid], places=self.places_points)
                    assert_allclose(self.syn_net.segment[sid].point[pid], self.net.segment[sid].point[pid],
                                    rtol=1e-4, atol=0)

        def test_vector_dict(self):
            for sid in random.sample(self.net.available_segments, 2):
                for vid in random.sample(range(len(self.net.segment[sid]) - 1), 3):
                    assert_allclose(self.syn_net.segment[sid].vector[vid], self.net.segment[sid].vector[vid],
                                    rtol=1e-4, atol=0)

        def test_bbox(self):
            assert_allclose(self.syn_net.bbox, self.net.bbox, rtol=1e-4, atol=0)

        def test_center(self):
            assert_allclose(self.syn_net.center, self.net.center, rtol=1e-4, atol=0)

        def test_size(self):
            assert_allclose(self.syn_net.bbox_size, self.net.bbox_size, rtol=1e-4, atol=0)

        def test_network_attributes(self):
            for attr, places in self.attribute_list_base + self.attribute_list_net:
                self.assertAlmostEqual(getattr(self.syn_net, attr), getattr(self.net, attr), places=places, msg=attr)

        def test_fiber_attributes(self):
            if len(self.net.fiber) >= 2:
                sample = random.sample(list(self.net.fiber), 2)
            else:
                sample = self.net.fiber.keys()
            for fid in sample:
                for attr, places in self.attribute_list_base:
                    self.assertAlmostEqual(getattr(self.syn_net.fiber[fid], attr), getattr(self.net.fiber[fid], attr),
                                           places=places, msg=attr)

        def test_segment_attributes(self):
            for sid in random.sample(list(self.net.segment), 2):
                for attr, places in self.attribute_list_base:
                    self.assertAlmostEqual(getattr(self.syn_net.segment[sid], attr), getattr(self.net.segment[sid], attr),
                                           places=places, msg=attr)

    class LoadClose(unittest.TestCase):
        cases = Path(__file__).absolute().resolve().parent / 'cases'
        syn_net = None
        net = None
        places_radius = 3
        places_points = 3
        attribute_list_base = [('length', 0.02), ('volume', 0.02), ('average_radius', 0.03),
                               ('cylinder_radius', 0.03), ('average_diameter', 0.03), ('number_of_fibers', 0)]

        @classmethod
        def setUpClass(cls):
            cls.syn_net = IO.load.synthetic_network()

        @classmethod
        def tearDownClass(cls):
            del cls.syn_net
            del cls.net

        def test_network_attributes(self):
            for attr, rtol in self.attribute_list_base:
                np.testing.assert_allclose(getattr(self.syn_net, attr), getattr(self.net, attr), rtol=rtol)

        def test_fiber_attributes(self):
            syn_fiber_length_list = sorted([fiber.length for fiber in self.syn_net.fiber.values()])
            fiber_length_list = sorted([fiber.length for fiber in self.net.fiber.values()])
            np.testing.assert_allclose(syn_fiber_length_list, fiber_length_list, rtol=0.03)
            syn_fiber_volume_list = sorted([fiber.length for fiber in self.syn_net.fiber.values()])
            fiber_volume_list = sorted([fiber.length for fiber in self.net.fiber.values()])
            np.testing.assert_allclose(syn_fiber_volume_list, fiber_volume_list, rtol=0.03)

        def test_size(self):
            assert_allclose(self.syn_net.bbox_size, self.net.bbox_size, rtol=0.01, atol=0)


class TestMV3D(Model.Load):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.syn_net.export(cls.cases / 'synthetic.mv3d')
        cls.net = Network.load(cls.cases / 'synthetic.mv3d')

    @classmethod
    def tearDownClass(cls):
        cls.net.input_file.unlink()
        super().tearDownClass()


class TestJSON(Model.Load):

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.syn_net.export(cls.cases / 'synthetic.json')
        cls.net = Network.load(cls.cases / 'synthetic.json')

    @classmethod
    def tearDownClass(cls):
        cls.net.input_file.unlink()
        super().tearDownClass()


class TestBinary(Model.Load):

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.syn_net.save(cls.cases / 'synthetic.qiber', overwrite=True)
        cls.net = Network.load(cls.cases / 'synthetic.qiber')

    @classmethod
    def tearDownClass(cls):
        cls.net.input_file.unlink()
        super().tearDownClass()


class TestImage(Model.LoadClose):

    @classmethod
    def setUpClass(cls):
        voxel_resolution = 5
        config.extract.voxel_size = [1 / voxel_resolution] * 3
        config.extract.morph.apply = False
        config.extract.median.apply = False
        config.extract.smooth.apply = False
        config.extract.z_drop.apply = False
        config.extract.binary.threshold = 29
        config.core_count = 1
        super().setUpClass()
        cls.syn_net.export(cls.cases / 'synthetic.tif', voxel_resolution=voxel_resolution, overwrite=True)
        cls.net = Network.load(cls.cases / 'synthetic.tif')

    @classmethod
    def tearDownClass(cls):
        try:
            cls.net.input_file.unlink()
        except PermissionError:
            pass
        super().tearDownClass()


class TestSWC(Model.Load):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.syn_net.export(cls.cases / 'synthetic.swc', overwrite=True)
        cls.syn_net = Network.load(cls.cases / 'result_swc_synthetic.json')
        cls.net = Network.load(cls.cases / 'synthetic.swc')

    @classmethod
    def tearDownClass(cls):
        try:
            cls.net.input_file.unlink()
        except PermissionError:
            pass
        super().tearDownClass()


class TestNTR(Model.Load):
    @classmethod
    def setUpClass(cls):
        cls.syn_net = Network.load(cls.cases / 'result_ntr_cb7b.json')
        cls.net = Network.load(cls.cases / 'neuron_cb7b.ntr')

    @classmethod
    def tearDownClass(cls):
        super().tearDownClass()
