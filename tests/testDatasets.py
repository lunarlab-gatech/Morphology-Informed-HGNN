import unittest
from pathlib import Path
from grfgnn import FlexibleDataset, QuadSDKDataset_A1Speed1_0, QuadSDKDataset, QuadSDKDataset_A1Speed0_5
from grfgnn.datasets_py.LinTzuYaunDataset import LinTzuYaunDataset, LinTzuYaunDataset_concrete_right_circle
from torch_geometric.data import Data, HeteroData
import numpy as np
import torch

class TestFlexibleDatasets(unittest.TestCase):
    """
    Test that the Flexible datasets general methods work for both 
    the QuadSDK dataset and the LinTzuYaun dataset, so we can be
    confident that it will work for any other dataset without more
    test cases.
    """

    def setUp(self):
        # Get the paths to the URDF files
        self.path_to_a1_urdf = Path(
            Path('.').parent, 'urdf_files', 'A1', 'a1.urdf').absolute()
        self.path_to_mc_urdf = Path('urdf_files', 'MiniCheetah', 'miniCheetah.urdf').absolute()
        
        # Set up the paths to the datasets
        self.path_to_quad_seq = Path(
            Path('.').parent, 'datasets', 'QuadSDK-A1Speed1.0').absolute()
        self.path_to_lin_seq = Path(Path('.').parent, 'datasets', 'LinTzuYaun-CRC').absolute()

        # Set up the QuadSDK datasets
        self.quad_dataset_hgnn_1 = QuadSDKDataset_A1Speed1_0(self.path_to_quad_seq,
            self.path_to_a1_urdf, 'package://a1_description/',
                            'unitree_ros/robots/a1_description', 'heterogeneous_gnn', 1)
        self.quad_dataset_mlp_1 = QuadSDKDataset_A1Speed1_0(self.path_to_quad_seq,
            self.path_to_a1_urdf, 'package://a1_description/',
                            'unitree_ros/robots/a1_description', 'mlp', 1)
        self.quad_dataset_hgnn_3 = QuadSDKDataset_A1Speed1_0(self.path_to_quad_seq,
            self.path_to_a1_urdf, 'package://a1_description/',
                            'unitree_ros/robots/a1_description', 'heterogeneous_gnn', 3)
        self.quad_dataset_mlp_3 = QuadSDKDataset_A1Speed1_0(self.path_to_quad_seq,
            self.path_to_a1_urdf, 'package://a1_description/',
                            'unitree_ros/robots/a1_description', 'mlp', 3)

        # Setup the the LinTzuYaun datasets
        self.lin_dataset_hgnn_1 = LinTzuYaunDataset_concrete_right_circle(
            self.path_to_lin_seq, self.path_to_mc_urdf, 'package://yobotics_description/',
            'mini-cheetah-gazebo-urdf/yobo_model/yobotics_description', 'heterogeneous_gnn', 1)
        self.lin_dataset_mlp_1 = LinTzuYaunDataset_concrete_right_circle(
            self.path_to_lin_seq, self.path_to_mc_urdf, 'package://yobotics_description/',
            'mini-cheetah-gazebo-urdf/yobo_model/yobotics_description', 'mlp', 1)
        self.lin_dataset_hgnn_3 = LinTzuYaunDataset_concrete_right_circle(
            self.path_to_lin_seq, self.path_to_mc_urdf, 'package://yobotics_description/',
            'mini-cheetah-gazebo-urdf/yobo_model/yobotics_description', 'heterogeneous_gnn', 3)
        self.lin_dataset_mlp_3 = LinTzuYaunDataset_concrete_right_circle(
            self.path_to_lin_seq, self.path_to_mc_urdf, 'package://yobotics_description/',
            'mini-cheetah-gazebo-urdf/yobo_model/yobotics_description', 'mlp', 3)
        
        # Set up datasets to test normalization
        self.quad_dataset_hgnn_6_norm = QuadSDKDataset_A1Speed1_0(self.path_to_quad_seq,
            self.path_to_a1_urdf, 'package://a1_description/',
            'unitree_ros/robots/a1_description', 'heterogeneous_gnn', 
            6, normalize=True)
        self.lin_dataset_hgnn_10_norm = LinTzuYaunDataset_concrete_right_circle(
            self.path_to_lin_seq, self.path_to_mc_urdf, 'package://yobotics_description/',
            'mini-cheetah-gazebo-urdf/yobo_model/yobotics_description', 'heterogeneous_gnn', 
            10, normalize=True)

    def test__init__(self):
        """
        Test that the __init__ method properly detect when the user
        erroneously gives a path to dataset folder that doesn't match
        the Dataset Sequence class created.

        Also test that parent classes without a specific sequence can't
        be created.
        """
        # ================== Test erroneous path checking ==================

        # Test the __init__ function properly runs on a new sequence
        path_to_slow_sequence = Path(
            Path('.').parent, 'datasets', 'QuadSDK-A1Speed0.5').absolute()
        dataset_slow = QuadSDKDataset_A1Speed0_5(path_to_slow_sequence,
                            self.path_to_a1_urdf, 'package://a1_description/',
                            'unitree_ros/robots/a1_description', 'heterogeneous_gnn', 1)
        
        # Try to create a normal sequence, pointing to the slow dataset directory
        with self.assertRaises(ValueError):
            dataset = QuadSDKDataset_A1Speed0_5(self.path_to_quad_seq,
                                 self.path_to_a1_urdf, 'package://a1_description/',
                                'unitree_ros/robots/a1_description', 'heterogeneous_gnn', 1)
            
        # Try to create a slow sequence, pointing to the normal dataset directory
        with self.assertRaises(ValueError) as e:
            dataset = QuadSDKDataset_A1Speed1_0(path_to_slow_sequence,
                                 self.path_to_a1_urdf, 'package://a1_description/',
                                'unitree_ros/robots/a1_description', 'heterogeneous_gnn', 1)

        # =========== Test non-sequence classes can't be created ===========

        # Make fake directories for testing purposes
        path_to_flexible = Path(Path('.').parent, 'datasets',
                                'test_case_dir', 'Flexible').absolute()
        path_to_quad_sdk = Path(Path('.').parent, 'datasets',
                                'test_case_dir', 'QuadSDK').absolute()
        path_to_lin = Path(Path('.').parent, 'datasets',
                                'test_case_dir', 'LinTzuYaun').absolute()

        # Try to create non-sequence classes, hope for an error
        with self.assertRaises(NotImplementedError):
            FlexibleDataset(path_to_flexible, self.path_to_a1_urdf, 'package://a1_description/',
                'unitree_ros/robots/a1_description', 'heterogeneous_gnn', 1)
        with self.assertRaises(NotImplementedError):
            QuadSDKDataset(path_to_quad_sdk, self.path_to_a1_urdf, 'package://a1_description/',
                'unitree_ros/robots/a1_description', 'heterogeneous_gnn', 1)
        with self.assertRaises(NotImplementedError):
            LinTzuYaunDataset(path_to_lin, self.path_to_mc_urdf, 'package://yobotics_description/',
                'mini-cheetah-gazebo-urdf/yobo_model/yobotics_description', 'heterogeneous_gnn', 1)
        
        
    def test_load_data_sorted(self):
        """
        Test that the sorting function matches the order provided by the child class.
        """
        # ====================== Test QuadSDK dataset ======================
        la, av, p, v, t, fp, fv, gt, r_p, r_o = self.quad_dataset_hgnn_1.load_data_sorted(10000)
        des_la = [[-0.06452160178213015, -0.366493877667443, 9.715652148737323]]
        des_av = [[-0.0017398309484803294, -0.011335050676391335, 1.2815129213608234]]
        des_p = [[-0.16056788963386381,  0.6448773402529877, 1.1609664103261004,
                 0.11254642823264671, 0.5695073200020913, 0.9825683053175158,
                 -0.1931352599922853,  0.4076711540455795,  0.9424138768126973,
                  0.22800622574234453, 0.4399285706508147, 1.1786520769378077]]
        des_v = [[-1.015061543404335, 0.459564643757568, 0.15804277355899754,
                 -1.2845189574751656, 2.2337115054710917, 3.4964483387811476,
                  1.9489188274516005, 1.3299772937985548, 3.644698146547278,
                  1.020374615076573, -0.34825271015763287, 0.3185087654826033]]
        des_t = [[-3.658930090797254, -3.858403899440098, 12.475195605359755,
                 -0.6701594400127251,-0.48841280756095506, -0.1350813273560049,
                 0.6111713969715354, -0.11888726638996594, -0.24871924601280232,
                 3.6351217639084576, 4.456326408036115, 7.829759255207876]]
        des_fp = None
        des_fv = None
        des_gt = [[0, 64.74924447333427, 64.98097097053076, 0]]
        des_rp = [[-0.9077062285177977, 0.400808137988427, 0.25139755534263925]]
        des_ro = [[0.00807844275531408, 0.006594002480070749, -0.6310441334749343, -0.7756768396057802]]
        np.testing.assert_array_equal(la, des_la)
        np.testing.assert_array_equal(av, des_av)
        np.testing.assert_array_equal(p, des_p)
        np.testing.assert_array_equal(v, des_v)
        np.testing.assert_array_equal(t, des_t)
        np.testing.assert_array_equal(fp, des_fp)
        np.testing.assert_array_equal(fv, des_fv)
        np.testing.assert_array_equal(gt, des_gt)
        np.testing.assert_array_almost_equal(r_p, des_rp, 16)
        np.testing.assert_array_almost_equal(r_o, des_ro, 16)

        # ===================== Test LinTzuYaun dataset ====================
        la, av, p, v, t, fp, fv, gt, r_p, r_o = self.lin_dataset_mlp_1.load_data_sorted(30000)
        des_la = [[0.146148949861527,	0.0223360173404217,	9.58438873291016]]
        des_av = [[0.00771053740754724,	0.0315540358424187,	0.161459520459175]]
        des_p = [[0.0581750869750977,	-0.498899102210999,	1.01744437217712,	
                 0.000190734863281250,	-0.625548958778381,	1.45497155189514,	
                 0.00858306884765625, -0.650104880332947,	1.49907255172730,	
                 0.0394830703735352,	-0.420838713645935,	0.846460103988648]]
        des_v = [[0.0793685913085938,	-1.38095092773438,	0.0102071752771735,	
                 0.142860412597656,	8.93650817871094,	-9.21489906311035,	
                 -1.60317611694336,	9.79365158081055,	-9.19449234008789,	
                 0.301589965820313,	-1.47618865966797,	-0.418396085500717]]
        des_t = None
        des_fp = [[-0.00335261225700378,	-0.0453704893589020,	-0.356130868196487,	
                  0.0214422345161438,	0.0660574287176132,	-0.301095396280289,	
                  0.0198653340339661,	-0.0634637400507927,	-0.295771747827530,	
                  -0.00486903637647629,	0.0804890319705010,	-0.365474313497543]]
        des_fv = [[-0.485595673322678,	0.0279390625655651,	0.00200577825307846,	
                  1.47740077972412,	0.0432308465242386,	-1.12423670291901,	
                  1.70655310153961,	-0.464293301105499,	-1.04942762851715,	
                  -0.618086814880371,	0.111269362270832, -0.00220335274934769]]
        des_gt = [[1, 0, 0, 1]]
        des_rp = None
        des_ro = None
        np.testing.assert_array_almost_equal(la, des_la, 14)
        np.testing.assert_array_almost_equal(av, des_av, 14)
        np.testing.assert_array_almost_equal(p, des_p, 14)
        np.testing.assert_array_almost_equal(v, des_v, 14)
        np.testing.assert_equal(t, des_t)
        np.testing.assert_array_almost_equal(fp, des_fp, 14)
        np.testing.assert_array_almost_equal(fv, des_fv, 14)
        np.testing.assert_array_almost_equal(gt, des_gt, 14)
        np.testing.assert_equal(r_p, des_rp)
        np.testing.assert_equal(r_o, des_ro)

    def test_get_helper_mlp(self):
        # ====================== Test QuadSDK dataset ======================
        x, y = self.quad_dataset_mlp_1.get_helper_mlp(10000)

        des_x = np.array([-0.06452160178213015, -0.366493877667443, 9.715652148737323,
                 -0.0017398309484803294, -0.011335050676391335, 1.2815129213608234,
                 -0.16056788963386381,  0.6448773402529877, 1.1609664103261004,
                 0.11254642823264671, 0.5695073200020913, 0.9825683053175158,
                 -0.1931352599922853,  0.4076711540455795,  0.9424138768126973,
                  0.22800622574234453, 0.4399285706508147, 1.1786520769378077,
                 -1.015061543404335, 0.459564643757568, 0.15804277355899754,
                 -1.2845189574751656, 2.2337115054710917, 3.4964483387811476,
                  1.9489188274516005, 1.3299772937985548, 3.644698146547278,
                  1.020374615076573, -0.34825271015763287, 0.3185087654826033,
                  -3.658930090797254, -3.858403899440098, 12.475195605359755,
                 -0.6701594400127251,-0.48841280756095506, -0.1350813273560049,
                 0.6111713969715354, -0.11888726638996594, -0.24871924601280232,
                 3.6351217639084576, 4.456326408036115, 7.829759255207876], dtype=np.float64)
        des_y = np.array([0, 64.74924447333427, 64.98097097053076, 0], dtype=np.float64)

        np.testing.assert_array_equal(y, des_y)
        np.testing.assert_array_equal(x, des_x)

        # ===================== Test LinTzuYaun dataset ====================
        x, y = self.lin_dataset_mlp_1.get_helper_mlp(30000)

        des_x = np.array([0.146148949861527,	0.0223360173404217,	9.58438873291016,
                  0.00771053740754724,	0.0315540358424187,	0.161459520459175,
                  0.0581750869750977,	-0.498899102210999,	1.01744437217712,	
                 0.000190734863281250,	-0.625548958778381,	1.45497155189514,	
                 0.00858306884765625, -0.650104880332947,	1.49907255172730,	
                 0.0394830703735352,	-0.420838713645935,	0.846460103988648,
                 0.0793685913085938,	-1.38095092773438,	0.0102071752771735,	
                 0.142860412597656,	8.93650817871094,	-9.21489906311035,	
                 -1.60317611694336,	9.79365158081055,	-9.19449234008789,	
                 0.301589965820313,	-1.47618865966797,	-0.418396085500717,
                 -0.00335261225700378,	-0.0453704893589020,	-0.356130868196487,	
                  0.0214422345161438,	0.0660574287176132,	-0.301095396280289,	
                  0.0198653340339661,	-0.0634637400507927,	-0.295771747827530,	
                  -0.00486903637647629,	0.0804890319705010,	-0.365474313497543,
                  -0.485595673322678,	0.0279390625655651,	0.00200577825307846,	
                  1.47740077972412,	0.0432308465242386,	-1.12423670291901,	
                  1.70655310153961,	-0.464293301105499,	-1.04942762851715,	
                  -0.618086814880371,	0.111269362270832, -0.00220335274934769], dtype=np.float64)
        des_y = np.array([1, 0,	0, 1], dtype=np.float64)

        np.testing.assert_array_equal(y, des_y)
        np.testing.assert_array_almost_equal(x, des_x, 14)

    def test_get_helper_heterogeneous_gnn(self):
        # ====================== Test QuadSDK dataset ======================
        heteroData: HeteroData = self.quad_dataset_hgnn_1.get_helper_heterogeneous_gnn(10000)

        # Test the desired edge matrices
        bj, jb, jj, fj, jf = self.quad_dataset_hgnn_1.robotGraph.get_edge_index_matrices()
        np.testing.assert_array_equal(
            heteroData['base', 'connect', 'joint'].edge_index.numpy(), bj)
        np.testing.assert_array_equal(
            heteroData['joint', 'connect', 'base'].edge_index.numpy(), jb)
        np.testing.assert_array_equal(
            heteroData['joint', 'connect', 'joint'].edge_index.numpy(), jj)
        np.testing.assert_array_equal(
            heteroData['foot', 'connect', 'joint'].edge_index.numpy(), fj)
        np.testing.assert_array_equal(
            heteroData['joint', 'connect', 'foot'].edge_index.numpy(), jf)

        # Check the edge attributes
        bj_attr, jb_attr, jj_attr, fj_attr, jf_attr = self.quad_dataset_hgnn_1.robotGraph.get_edge_attr_matrices()
        np.testing.assert_array_equal(heteroData['base', 'connect', 'joint'].edge_attr.numpy(), bj_attr)
        np.testing.assert_array_equal(heteroData['joint', 'connect', 'base'].edge_attr.numpy(), jb_attr)
        np.testing.assert_array_equal(heteroData['joint', 'connect', 'joint'].edge_attr.numpy(), jj_attr)
        np.testing.assert_array_equal(heteroData['foot', 'connect', 'joint'].edge_attr.numpy(), fj_attr)
        np.testing.assert_array_equal(heteroData['joint', 'connect', 'foot'].edge_attr.numpy(), jf_attr)

        # Check the labels
        labels_des = [0, 64.74924447333427, 64.98097097053076, 0]
        np.testing.assert_array_equal(heteroData.y.numpy(),
                                      np.array(labels_des, dtype=np.float64))

        # Check the number of nodes
        number_des = self.quad_dataset_hgnn_1.robotGraph.get_num_nodes()
        self.assertEqual(heteroData.num_nodes, number_des)

        # Check the node attributes
        base_x = np.array([[-0.06452160178213015, -0.366493877667443, 9.715652148737323,
                            -0.0017398309484803294, -0.011335050676391335, 1.2815129213608234]], dtype=np.float64)
        joint_x = np.array([[0.11254642823264671, -1.2845189574751656, -0.6701594400127251],
                            [0.5695073200020913,  2.2337115054710917, -0.48841280756095506],
                            [0.9825683053175158, 3.4964483387811476, -0.1350813273560049],
                            [-0.16056788963386381, -1.015061543404335, -3.658930090797254],
                            [ 0.6448773402529877, 0.459564643757568, -3.858403899440098],
                            [1.1609664103261004, 0.15804277355899754, 12.475195605359755],
                            [0.22800622574234453, 1.020374615076573, 3.6351217639084576],
                            [0.4399285706508147, -0.34825271015763287, 4.456326408036115],
                            [1.1786520769378077, 0.3185087654826033,  7.829759255207876],
                            [-0.1931352599922853, 1.9489188274516005, 0.6111713969715354],
                            [ 0.4076711540455795, 1.3299772937985548,  -0.11888726638996594],
                            [ 0.9424138768126973, 3.644698146547278, -0.24871924601280232]], dtype=np.float64)
        foot_x = np.array([[1], [1], [1], [1]], dtype=np.float32)
        np.testing.assert_array_equal(heteroData['base'].x.numpy(), base_x)
        np.testing.assert_array_equal(heteroData['joint'].x.numpy(), joint_x)
        np.testing.assert_array_equal(heteroData['foot'].x.numpy(), foot_x)

        # ===================== Test LinTzuYaun dataset ====================
        heteroData: HeteroData = self.lin_dataset_hgnn_1.get_helper_heterogeneous_gnn(30000)

        # Test the desired edge matrices
        bj, jb, jj, fj, jf = self.lin_dataset_hgnn_1.robotGraph.get_edge_index_matrices()
        np.testing.assert_array_equal(
            heteroData['base', 'connect', 'joint'].edge_index.numpy(), bj)
        np.testing.assert_array_equal(
            heteroData['joint', 'connect', 'base'].edge_index.numpy(), jb)
        np.testing.assert_array_equal(
            heteroData['joint', 'connect', 'joint'].edge_index.numpy(), jj)
        np.testing.assert_array_equal(
            heteroData['foot', 'connect', 'joint'].edge_index.numpy(), fj)
        np.testing.assert_array_equal(
            heteroData['joint', 'connect', 'foot'].edge_index.numpy(), jf)
        
        # Check the edge attributes
        bj_attr, jb_attr, jj_attr, fj_attr, jf_attr = self.lin_dataset_hgnn_1.robotGraph.get_edge_attr_matrices()
        np.testing.assert_array_equal(heteroData['base', 'connect', 'joint'].edge_attr.numpy(), bj_attr)
        np.testing.assert_array_equal(heteroData['joint', 'connect', 'base'].edge_attr.numpy(), jb_attr)
        np.testing.assert_array_equal(heteroData['joint', 'connect', 'joint'].edge_attr.numpy(), jj_attr)
        np.testing.assert_array_equal(heteroData['foot', 'connect', 'joint'].edge_attr.numpy(), fj_attr)
        np.testing.assert_array_equal(heteroData['joint', 'connect', 'foot'].edge_attr.numpy(), jf_attr)

        # Check the labels
        labels_des = [1, 0,	0, 1]
        np.testing.assert_array_equal(heteroData.y.numpy(), np.array(labels_des, dtype=np.float64))

        # Check the number of nodes
        number_des = self.lin_dataset_hgnn_1.robotGraph.get_num_nodes()
        self.assertEqual(heteroData.num_nodes, number_des)

         # Check the node attributes
        base_x = np.array([[0.146148949861527,	0.0223360173404217,	9.58438873291016,
                            0.00771053740754724,	0.0315540358424187,	0.161459520459175]], dtype=np.float64)
        joint_x = np.array([[0.0394830703735352, 0.301589965820313],
                            [-0.420838713645935,-1.47618865966797],
                            [0.846460103988648,-0.418396085500717],
                            [0.000190734863281250,0.142860412597656],
                            [-0.625548958778381,8.93650817871094],
                            [1.45497155189514,-9.21489906311035],
                            [0.00858306884765625,-1.60317611694336],
                            [-0.650104880332947,9.79365158081055],
                            [1.49907255172730,-9.19449234008789],
                            [0.0581750869750977,0.0793685913085938],
                            [-0.498899102210999,-1.38095092773438],
                            [1.01744437217712,0.0102071752771735],], dtype=np.float64)
        foot_x = np.array([[-0.00486903637647629, 0.0804890319705010, -0.365474313497543, -0.618086814880371,	0.111269362270832, -0.00220335274934769], 
                           [0.0214422345161438,	0.0660574287176132,	-0.301095396280289, 1.47740077972412,	0.0432308465242386,	-1.12423670291901], 
                           [0.0198653340339661,	-0.0634637400507927,	-0.295771747827530, 1.70655310153961,	-0.464293301105499,	-1.04942762851715], 
                           [-0.00335261225700378, -0.0453704893589020, -0.356130868196487, -0.485595673322678,	0.0279390625655651,	0.00200577825307846]], dtype=np.float32)
        np.testing.assert_array_almost_equal(heteroData['base'].x.numpy(), base_x, 14)
        np.testing.assert_array_almost_equal(heteroData['joint'].x.numpy(), joint_x, 14)
        np.testing.assert_array_equal(heteroData['foot'].x.numpy(), foot_x)

    def test_get(self):
        # Test that the get function reacts properly to dataset bounds.
        with self.assertRaises(IndexError):
            data = self.quad_dataset_hgnn_1.get(-1)
        with self.assertRaises(IndexError):
            data = self.quad_dataset_hgnn_1.get(17531)
        data = self.quad_dataset_hgnn_1.get(0)
        data = self.quad_dataset_hgnn_1.get(17530)

        # Test that it returns the proper values based on the model type
        x, y = self.quad_dataset_mlp_1.get(0)
        self.assertEqual(type(x), torch.Tensor)
        self.assertEqual(type(y), torch.Tensor)

        data = self.quad_dataset_hgnn_1.get(0)
        self.assertEqual(type(data), HeteroData)

    def test_history_length_parameter(self):
        """
        This helper method tests that changing the history lengths
        causes the get method to return inputs with proper history
        data.
        """

        # Make sure get method doesn't step out of bounds (since our dataset
        # length has been altered)
        with self.assertRaises(IndexError):
            data = self.quad_dataset_mlp_3.get(-1)
        with self.assertRaises(IndexError):
            data = self.quad_dataset_mlp_3.get(17529)
        data = self.quad_dataset_mlp_3.get(0)
        data = self.quad_dataset_mlp_3.get(17528)

        # ================================= MLP ==========================================
        mlp_3_datasets = [self.quad_dataset_mlp_3, self.lin_dataset_mlp_3]
        mlp_1_datasets = [self.quad_dataset_mlp_1, self.lin_dataset_mlp_1]
        for i in range(0, 2):
            x_actual, y_actual = mlp_3_datasets[i].get(9998)

            # Calculated the desired x and y values
            xb2, yb2 = mlp_1_datasets[i].get_helper_mlp(9998)
            xb1, yb1 = mlp_1_datasets[i].get_helper_mlp(9999)
            x, y_des = mlp_1_datasets[i].get_helper_mlp(10000)
            x_comb = torch.stack((xb2, xb1, x), 0)
            x_des = torch.flatten(torch.transpose(x_comb, 0, 1), 0, 1)

            # Test the values
            np.testing.assert_equal(x_actual.numpy(), x_des.numpy())
            np.testing.assert_equal(y_actual.numpy(), y_des.numpy())

        # ================================= Heterogeneous GNN ==========================================
        hgnn_3_datasets = [self.quad_dataset_hgnn_3, self.lin_dataset_hgnn_3]
        hgnn_1_datasets = [self.quad_dataset_hgnn_1, self.lin_dataset_hgnn_1]
        for i in range(0, 2):
            # Get the HeteroData graph
            heteroData: HeteroData = hgnn_3_datasets[i].get_helper_heterogeneous_gnn(9998)

            # Get the desired node attributes
            hDatab2 = hgnn_1_datasets[i].get_helper_heterogeneous_gnn(9998)
            hDatab1 = hgnn_1_datasets[i].get_helper_heterogeneous_gnn(9999)
            hData = hgnn_1_datasets[i].get_helper_heterogeneous_gnn(10000)
            base_x_cat = torch.cat((hDatab2['base'].x, hDatab1['base'].x, hData['base'].x), 0)
            base_x_des = torch.flatten(torch.transpose(base_x_cat, 0, 1), 0).unsqueeze(0)
            joint_x_des = None
            if i == 0:
                joint_x_des = torch.cat((hDatab2['joint'].x[:,0].unsqueeze(1), hDatab1['joint'].x[:,0].unsqueeze(1), hData['joint'].x[:,0].unsqueeze(1),
                                        hDatab2['joint'].x[:,1].unsqueeze(1), hDatab1['joint'].x[:,1].unsqueeze(1), hData['joint'].x[:,1].unsqueeze(1),
                                        hDatab2['joint'].x[:,2].unsqueeze(1), hDatab1['joint'].x[:,2].unsqueeze(1), hData['joint'].x[:,2].unsqueeze(1)), 1)
            elif i == 1:
                joint_x_des = torch.cat((hDatab2['joint'].x[:,0].unsqueeze(1), hDatab1['joint'].x[:,0].unsqueeze(1), hData['joint'].x[:,0].unsqueeze(1),
                                        hDatab2['joint'].x[:,1].unsqueeze(1), hDatab1['joint'].x[:,1].unsqueeze(1), hData['joint'].x[:,1].unsqueeze(1)), 1)
            else:
                raise NotImplementedError
            
            la, av, p, v, t, fp, fv, gt, r_p, r_o = hgnn_1_datasets[i].load_data_sorted(10000)
            foot_x_des = None
            if fp is None and fv is None:
                foot_x_des = hData['foot'].x
            else:
                foot_x_des = torch.cat((hDatab2['foot'].x[:,0].unsqueeze(1), hDatab1['foot'].x[:,0].unsqueeze(1), hData['foot'].x[:,0].unsqueeze(1),
                                        hDatab2['foot'].x[:,1].unsqueeze(1), hDatab1['foot'].x[:,1].unsqueeze(1), hData['foot'].x[:,1].unsqueeze(1),
                                        hDatab2['foot'].x[:,2].unsqueeze(1), hDatab1['foot'].x[:,2].unsqueeze(1), hData['foot'].x[:,2].unsqueeze(1),
                                        hDatab2['foot'].x[:,3].unsqueeze(1), hDatab1['foot'].x[:,3].unsqueeze(1), hData['foot'].x[:,3].unsqueeze(1),
                                        hDatab2['foot'].x[:,4].unsqueeze(1), hDatab1['foot'].x[:,4].unsqueeze(1), hData['foot'].x[:,4].unsqueeze(1),
                                        hDatab2['foot'].x[:,5].unsqueeze(1), hDatab1['foot'].x[:,5].unsqueeze(1), hData['foot'].x[:,5].unsqueeze(1)), 1)
            y = hData.y

            # Check the values
            np.testing.assert_array_almost_equal(heteroData['base'].x.numpy(), base_x_des.numpy(), 12)
            np.testing.assert_array_almost_equal(heteroData['joint'].x.numpy(), joint_x_des.numpy(), 6)
            np.testing.assert_array_equal(heteroData['foot'].x.numpy(), foot_x_des.numpy())
            np.testing.assert_array_equal(heteroData.y.numpy(), y.numpy())

    def test_normalize_parameter(self):
        """
        This helper method tests that setting the normalize parameter
        properly normalizes the data from the load_data_sorted() method.
        """
        # ====================== Test QuadSDK dataset ======================
        la, av, p, v, t, fp, fv, gt, r_p, r_o = self.quad_dataset_hgnn_6_norm.load_data_sorted(10000)

        des_la = [[-0.5874105616686414, -1.8220316435037871, -1.7415432647742404],
				  [0.0464140104789931, -0.2473371978906197, -0.3608975879330433],
				  [0.0464140104789931, -0.2473371978906197, -0.3608975879330433],
				  [-0.0298239446162162, 0.0672719762513247, 0.4145668989644065],
				  [-1.3979548466015099, 1.2327609818176768, 1.4912555547500135],
				  [1.9223613319283970, 1.0166730812160218, 0.5575159869259070]]
        des_av = [[0.4296442077515022, -2.1813002991742905, -1.7572698436375411],
				  [-0.9887532470515812, 0.2103304104471665, -0.3010879838906267],
				  [-0.9887532470515812, 0.2103304104471665, -0.3010879838906267],
				  [-0.8306289870465896, 0.8607070485078034, 0.0425341307001678],
				  [0.7759751413225284, 0.3879362653633671, 1.0087131808464069],
				  [1.6025161320757222, 0.5119961644087900, 1.3081984998721530]]
        des_p = [[1.4516093091930675, -1.4561700914674851, 1.4857734803828633, -1.4742892095976563, -1.4355859582502075, -1.4599655229104807, 1.4691737547623922, -1.4749557080186593, -1.4601490461454085, -1.4678477983513907, 1.4670557418591046, 1.4628143320732248],
				  [0.8781269562513216, -0.8790215277506471, 0.8788268724841540, -0.8780437053770196, -0.8845900633286349, -0.8796245236239294, 0.8752550065973494, -0.8768906459067572, -0.8801082266402102, -0.8802532935734834, 0.8763166043351074, 0.8503212722357105],
				  [0.3016069430968103, -0.2979641478617038, 0.2762063873499779, -0.2853054623643357, -0.3146964891889050, -0.2955871551441219, 0.2872761516411941, -0.2837454749806904, -0.2956704920644266, -0.2903488590363837, 0.2903511396554483, 0.2909687019925287],
				  [-0.2795648867927041, 0.2853603779529442, -0.3170920146836194, 0.3040663472289860, 0.2714335108636111, 0.2903466755920730, -0.2940181040778332, 0.3028026065411095, 0.2912507371393076, 0.2996215202933490, -0.2937055932466995, -0.2564504418110940],
				  [-0.8699728726562852, 0.8733939132661065, -0.8940158833695038, 0.8849387002871872, 0.8727199918023725, 0.8774802591820905, -0.8738068723085045, 0.8814255221455758, 0.8787647935210735, 0.8852995812414083, -0.8772564650223248, -0.8340041729421219],
				  [-1.4818054490922101, 1.4744014758605650, -1.4296988421638721, 1.4486333298228471, 1.4907190081017863, 1.4673502669043508, -1.4638799366146045, 1.4513637002194515, 1.4659122341896642, 1.4535288494265155, -1.4627614275806360, -1.5136496915519828]]
        des_v = [[0.7400296844223826, -0.9959130795984620, -0.8715558132444908, 0.8936484209897618, -1.4958874653393361, -1.5360207861860993, -1.6907582916534534, 1.3142155325429250, -1.8223879926204389, -0.0133888324122987, -1.8765446739277087, -1.4798463281105914],
				  [0.8411389137875830, -0.9445858223055260, -0.9276007343777303, 0.8292763669108429, -0.8965468037856165, -0.9816984180693528, -0.3700880580564146, 0.8573909121030795, -0.8481559362963309, 0.6606488174658279, -0.7308234777973001, -0.1696005931222993],
				  [0.5892862815911963, -0.4282679170708277, -0.5759963903772310, 0.5027858414795752, -0.2182767572588236, 0.0464020251585243, 0.2445062414529919, 0.4480146751851610, 0.3138928677081939, 0.7486915524336630, 0.3359455079592579, 0.8317778996904992],
				  [0.2808194524348734, -0.0653863170428016, -0.0831373990309138, 0.2665513000509425, 0.3178200938496814, 0.3608039725145944, 1.1660792426064868, -0.1582202929211012, 0.7955922522905156, 0.6885461624258753, 0.7460383092597320, 1.1380405633119672],
				  [-0.4149736781622943, 0.5222113875084845, 0.5151185644702069, -0.5098683598833530, 0.8454506568434648, 0.6568581166826144, 1.1538123771330651, -0.8547213367527147, 0.8497588971812162, 0.0431459377127387, 0.9989609801277612, 0.7733507103105561],
				  [-2.0363006540737079, 1.9119417485090799, 1.9431717725601507, -1.9823935695477692, 1.4474402756906271, 1.4536550898997367, -0.5035515114826757, -1.6066794901573598, 0.7112999117367694, -2.1276436376258125, 0.5264233543782688, -1.0937222520801908]]
        des_t = [[0.6650296668666529, 1.3549330139371512, -1.3016680376372194, -0.6628628823968141, 0.9850119712952055, 0.4761803514019395, -0.4875704258981984, 0.9138951607742647, 0.9556807283698490, 1.2648619105318490, -1.5920338472135789, 1.1224558005351830],
				  [0.4744912222247263, 1.0946338296014217, -0.7641950182576893, -0.8051851969730056, 1.5689107379807403, 1.6600210898540557, -0.4079414082154866, 1.3735873468717481, 1.4319867786123528, 0.8736948463378806, -0.8848246685376849, 0.9847509744726390],
				  [0.4253489970568188, -0.2457219322383162, -0.2996443893050191, 0.7250634288623693, -0.2896026495074269, -0.7370213196512883, 1.5415620602385749, 0.3030608440411146, 0.2438373926168658, 0.3450351302611718, -0.1022414347070203, 0.3876751674803817],
				  [0.7506895664437274, -0.0550938438409042, -0.1406238875130855, -0.4605852325506335, -0.5240783856019928, -0.5650979153048762, -0.9337754544020441, -0.5393612424833809, -0.4978470608330381, 0.0288337866031798, 0.3217969206302563, 0.0758423634466288],
				  [-0.1812477917156566, -0.5294199702800675, 0.7673922536193011, -0.7116349520496456, -0.2804081436706222, 0.5335375656731959, -0.9299427217925544, -0.3964555413838193, -0.5911546848381205, -0.8315013751750765, 1.0266748461606023, -0.8916600663688480],
				  [-2.1343116608762647, -1.6193310971792851, 1.7387390790937032, 1.9152048351077364, -1.4598335304959067, -1.3676197719730276, 1.2176679500696925, -1.6547265678199365, -1.5425031539279090, -1.6809242985590049, 1.2306281836674082, -1.6790642395659250]]
        des_fp = None
        des_fv = None
        des_gt = [[0, 66.058643452423740, 65.599211397640180, 0]]
        des_rp = [[1.4643146618282410, -1.4684417536230334, -1.4684190946446067],
				  [0.8733864801350466, -0.8771598969412419, -0.8766100273305281],
				  [0.2909392032948933, -0.2890730668259753, -0.2887370542507244],
				  [-0.2866695923627322, 0.2961402092879633, 0.2954867508841712],
				  [-0.8678846930403481, 0.8788922834522276, 0.8768698198246120],
				  [-1.4740860598551007, 1.4596422246500598, 1.4614096055185775]]
        des_ro = [[-1.4115791281884251, -0.3799799758306283, 1.4615733584770301, -1.4591484827719403],
				  [-0.4424153868784342, 0.2403468535372326, 0.8788281781339352, -0.8793462237288245],
				  [0.5146745798986114, 0.7381396927128603, 0.2945979150793995, -0.2965451755451923],
				  [1.1907499628327165, 0.9777633576401984, -0.2910404879817826, 0.2891569945943370],
				  [1.1033382099604403, 0.4477177325746334, -0.8779556374818784, 0.8775587643459891],
				  [-0.9547682376249088, -2.0239876606339933, -1.4660033262267684, 1.4683241231062649]]
        np.testing.assert_array_equal(la, des_la)
        np.testing.assert_array_equal(av, des_av)
        np.testing.assert_array_equal(p, des_p)
        np.testing.assert_array_equal(v, des_v)
        np.testing.assert_array_equal(t, des_t)
        np.testing.assert_array_equal(fp, des_fp)
        np.testing.assert_array_equal(fv, des_fv)
        np.testing.assert_array_equal(gt, des_gt)
        np.testing.assert_array_almost_equal(r_p, des_rp, 16)
        np.testing.assert_array_almost_equal(r_o, des_ro, 16)

        # ===================== Test LinTzuYaun dataset ====================
        la, av, p, v, t, fp, fv, gt, r_p, r_o = self.lin_dataset_hgnn_10_norm.load_data_sorted(30000)
        des_la = [[1.5275252316519465, -1.5275252316519465, -1.5275252316519483],
				  [1.5275252316519465, -1.5275252316519465, -1.5275252316519483],
				  [1.5275252316519465, -1.5275252316519465, -1.5275252316519483],
				  [-0.6546536707079771, 0.6546536707079770, 0.6546536707079754],
				  [-0.6546536707079771, 0.6546536707079770, 0.6546536707079754],
				  [-0.6546536707079771, 0.6546536707079770, 0.6546536707079754],
				  [-0.6546536707079771, 0.6546536707079770, 0.6546536707079754],
				  [-0.6546536707079771, 0.6546536707079770, 0.6546536707079754],
				  [-0.6546536707079771, 0.6546536707079770, 0.6546536707079754],
				  [-0.6546536707079771, 0.6546536707079770, 0.6546536707079754]]
        des_av = [[-1.5275252316519461, 1.5275252316519470, -1.5275252316519514],
				  [-1.5275252316519461, 1.5275252316519470, -1.5275252316519514],
				  [-1.5275252316519461, 1.5275252316519470, -1.5275252316519514],
				  [0.6546536707079770, -0.6546536707079772, 0.6546536707079724],
				  [0.6546536707079770, -0.6546536707079772, 0.6546536707079724],
				  [0.6546536707079770, -0.6546536707079772, 0.6546536707079724],
				  [0.6546536707079770, -0.6546536707079772, 0.6546536707079724],
				  [0.6546536707079770, -0.6546536707079772, 0.6546536707079724],
				  [0.6546536707079770, -0.6546536707079772, 0.6546536707079724],
				  [0.6546536707079770, -0.6546536707079772, 0.6546536707079724]]
        des_p = [[0.0000000000000000, 1.6482209935357053, 1.9999999999995473, -0.6546536707079772, -1.5714285714285714, 1.5635412199875935, 1.6367836342540927, -1.5948400373283673, 1.5562623635687995, -1.7320508075688774, 1.4709062282223535, 1.5713931419369040],
				  [0.0000000000000000, 1.6482209935357053, 1.9999999999995473, -0.6546536707079772, -1.5714285714285714, 1.5635412199875935, 1.6367836342540927, -1.5948400373283673, 1.5562623635687995, -1.7320508075688774, 1.4709062282223535, 1.5713931419369040],
				  [0.0000000000000000, 0.0784867139778907, -0.5000000000004527, -0.6546536707079772, -0.1428571428571428, 0.1491388009580618, 0.0884747910407618, -0.1238710708604557, 0.1548865190872633, 0.0000000000000000, 0.2190711403735420, 0.1428854858763529],
				  [0.0000000000000000, 0.0784867139778907, -0.5000000000004527, -0.6546536707079772, -0.1428571428571428, 0.1491388009580618, 0.0884747910407618, -0.1238710708604557, 0.1548865190872633, 0.0000000000000000, 0.2190711403735420, 0.1428854858763529],
				  [0.0000000000000000, 0.0784867139778907, -0.5000000000004527, -0.6546536707079772, -0.1428571428571428, 0.1491388009580618, 0.0884747910407618, -0.1238710708604557, 0.1548865190872633, 0.0000000000000000, 0.2190711403735420, 0.1428854858763529],
				  [0.0000000000000000, 0.0784867139778907, -0.5000000000004527, -0.6546536707079772, -0.1428571428571428, 0.1491388009580618, 0.0884747910407618, -0.1238710708604557, 0.1548865190872633, 0.0000000000000000, 0.2190711403735420, 0.1428854858763529],
				  [0.0000000000000000, 0.0784867139778907, -0.5000000000004527, -0.6546536707079772, -0.1428571428571428, 0.1491388009580618, 0.0884747910407618, -0.1238710708604557, 0.1548865190872633, 0.0000000000000000, 0.2190711403735420, 0.1428854858763529],
				  [0.0000000000000000, -1.2296251856536213, -0.5000000000004527, 1.5275252316519468, 1.2857142857142858, -1.2909254815884832, -1.2386470745706648, 1.2696784763196711, -1.2956524408579719, 1.1547005383792515, -1.3457227194374723, -1.2857379044186681],
				  [0.0000000000000000, -1.2296251856536213, -0.5000000000004527, 1.5275252316519468, 1.2857142857142858, -1.2909254815884832, -1.2386470745706648, 1.2696784763196711, -1.2956524408579719, 1.1547005383792515, -1.3457227194374723, -1.2857379044186681],
				  [0.0000000000000000, -1.2296251856536213, -0.5000000000004527, 1.5275252316519468, 1.2857142857142858, -1.2909254815884832, -1.2386470745706648, 1.2696784763196711, -1.2956524408579719, 1.1547005383792515, -1.3457227194374723, -1.2857379044186681]]
        des_v = [[1.0000000000000000, 1.3256979607197410, 1.9163791387719977, 1.0000000000000000, -0.0769230769230726, 1.5118633795906529, 0.6546536707079786, -0.5203059023730081, 1.7820848532219813, -1.7320508075688772, -1.4411533842457847, 1.6644793241918201],
				  [1.0000000000000000, 1.3256979607197410, 1.9163791387719977, 1.0000000000000000, -0.0769230769230726, 1.5118633795906529, 0.6546536707079786, -0.5203059023730081, 1.7820848532219813, -1.7320508075688772, -1.4411533842457847, 1.6644793241918201],
				  [-1.0000000000000000, -0.9798637100971992, -0.7268942234330928, -1.0000000000000000, -0.8461538461538418, 0.1889781208276896, 0.6546536707079786, 0.9662823901213246, -0.0524149357107511, 0.8660254037844386, 0.9607689228305222, -0.8962582055045945],
				  [-1.0000000000000000, -0.9798637100971992, -0.7268942234330928, -1.0000000000000000, -0.8461538461538418, 0.1889781208276896, 0.6546536707079786, 0.9662823901213246, -0.0524149357107511, 0.8660254037844386, 0.9607689228305222, -0.8962582055045945],
				  [-1.0000000000000000, -0.9798637100971992, -0.7268942234330928, -1.0000000000000000, -0.8461538461538418, 0.1889781208276896, 0.6546536707079786, 0.9662823901213246, -0.0524149357107511, 0.8660254037844386, 0.9607689228305222, -0.8962582055045945],
				  [-1.0000000000000000, -0.9798637100971992, -0.7268942234330928, -1.0000000000000000, -0.8461538461538418, 0.1889781208276896, 0.6546536707079786, 0.9662823901213246, -0.0524149357107511, 0.8660254037844386, 0.9607689228305222, -0.8962582055045945],
				  [-1.0000000000000000, -0.9798637100971992, -0.7268942234330928, -1.0000000000000000, -0.8461538461538418, 0.1889781208276896, 0.6546536707079786, 0.9662823901213246, -0.0524149357107511, 0.8660254037844386, 0.9607689228305222, -0.8962582055045945],
				  [1.0000000000000000, 0.7493075430155060, -0.0660957201261770, 1.0000000000000000, 1.4615384615384659, -1.3228724544399124, -1.5275252316519450, -1.2636000486201744, -1.1006983426300729, -0.2886751345948129, -0.6405126152203491, 0.3841107930464429],
				  [1.0000000000000000, 0.7493075430155060, -0.0660957201261770, 1.0000000000000000, 1.4615384615384659, -1.3228724544399124, -1.5275252316519450, -1.2636000486201744, -1.1006983426300729, -0.2886751345948129, -0.6405126152203491, 0.3841107930464429],
				  [1.0000000000000000, 0.7493075430155060, -0.0660957201261770, 1.0000000000000000, 1.4615384615384659, -1.3228724544399124, -1.5275252316519450, -1.2636000486201744, -1.1006983426300729, -0.2886751345948129, -0.6405126152203491, 0.3841107930464429]]
        des_t = None
        des_fp = [[1.6681546080045699, -1.5640574366689186, 1.5645711300947915, -1.5743247346855040, -0.6588378198124285, 1.5760072944262735, -1.6195476295826339, 1.6331601850660600, 1.5500570371017799, 1.4884212771470557, -1.7313631353725443, -1.8960217446096013],
				  [1.6681546080045699, -1.5640574366689186, 1.5645711300947915, -1.5743247346855040, -0.6588378198124285, 1.5760072944262735, -1.6195476295826339, 1.6331601850660600, 1.5500570371017799, 1.4884212771470557, -1.7313631353725443, -1.8960217446096013],
				  [0.0607021649814652, 0.9307581064966284, -0.9306073050248997, -0.1405363647947962, -0.6529777629653040, 0.1391845312301769, -0.1032467134746193, 0.0916076576280529, 0.1597499333908100, 0.2063520206057176, -0.0006872631459814, 0.1984032962922470],
				  [0.0607021649814652, 0.9307581064966284, -0.9306073050248997, -0.1405363647947962, -0.6529777629653040, 0.1391845312301769, -0.1032467134746193, 0.0916076576280529, 0.1597499333908100, 0.2063520206057176, -0.0006872631459814, 0.1984032962922470],
				  [0.0607021649814652, 0.9307581064966284, -0.9306073050248997, -0.1405363647947962, -0.6529777629653040, 0.1391845312301769, -0.1032467134746193, 0.0916076576280529, 0.1597499333908100, 0.2063520206057176, -0.0006872631459814, 0.1984032962922470],
				  [0.0607021649814652, 0.9307581064966284, -0.9306073050248997, -0.1405363647947962, -0.6529777629653040, 0.1391845312301769, -0.1032467134746193, 0.0916076576280529, 0.1597499333908100, 0.2063520206057176, -0.0006872631459814, 0.1984032962922470],
				  [0.0607021649814652, 0.9307581064966284, -0.9306073050248997, -0.1405363647947962, -0.6529777629653040, 0.1391845312301769, -0.1032467134746193, 0.0916076576280529, 0.1597499333908100, 0.2063520206057176, -0.0006872631459814, 0.1984032962922470],
				  [-1.2132733469721564, -0.5085585530219564, 0.5079647549844650, 1.2837770977816663, 1.5275214848171255, -1.2826457483344771, 1.2517762755127866, -1.2414528860907947, -1.2996212470524711, -1.3362008857742342, 1.1553875288249604, 0.9333423359155669],
				  [-1.2132733469721564, -0.5085585530219564, 0.5079647549844650, 1.2837770977816663, 1.5275214848171255, -1.2826457483344771, 1.2517762755127866, -1.2414528860907947, -1.2996212470524711, -1.3362008857742342, 1.1553875288249604, 0.9333423359155669],
				  [-1.2132733469721564, -0.5085585530219564, 0.5079647549844650, 1.2837770977816663, 1.5275214848171255, -1.2826457483344771, 1.2517762755127866, -1.2414528860907947, -1.2996212470524711, -1.3362008857742342, 1.1553875288249604, 0.9333423359155669]]
        des_fv = [[1.4893193892827439, 0.9903054932030352, 1.5988861876347582, 1.4653189984293840, 0.9423900132903743, 1.7289914708003520, 0.9640777403983550, 0.9587919646821867, 1.8540003574302877, -1.3774082988004341, -1.7314774277876119, -1.8408694731530220],
				  [1.4893193892827439, 0.9903054932030352, 1.5988861876347582, 1.4653189984293840, 0.9423900132903743, 1.7289914708003520, 0.9640777403983550, 0.9587919646821867, 1.8540003574302877, -1.3774082988004341, -1.7314774277876119, -1.8408694731530220],
				  [-0.9503535962985835, -0.9999843863825774, 0.1205361342559165, -0.9557426300721859, -0.9994570776453097, 0.0030512738177782, 0.5177485766884827, 0.5203247842029390, -0.1386878335903885, 0.9722572792885026, 0.8663118092391583, 0.7987434703505584],
				  [-0.9503535962985835, -0.9999843863825774, 0.1205361342559165, -0.9557426300721859, -0.9994570776453097, 0.0030512738177782, 0.5177485766884827, 0.5203247842029390, -0.1386878335903885, 0.9722572792885026, 0.8663118092391583, 0.7987434703505584],
				  [-0.9503535962985835, -0.9999843863825774, 0.1205361342559165, -0.9557426300721859, -0.9994570776453097, 0.0030512738177782, 0.5177485766884827, 0.5203247842029390, -0.1386878335903885, 0.9722572792885026, 0.8663118092391583, 0.7987434703505584],
				  [-0.9503535962985835, -0.9999843863825774, 0.1205361342559165, -0.9557426300721859, -0.9994570776453097, 0.0030512738177782, 0.5177485766884827, 0.5203247842029390, -0.1386878335903885, 0.9722572792885026, 0.8663118092391583, 0.7987434703505584],
				  [-0.9503535962985835, -0.9999843863825774, 0.1205361342559165, -0.9557426300721859, -0.9994570776453097, 0.0030512738177782, 0.5177485766884827, 0.5203247842029390, -0.1386878335903885, 0.9722572792885026, 0.8663118092391583, 0.7987434703505584],
				  [0.5910430676424782, 1.0064369818356056, -1.2668176821830330, 0.6160250511673788, 1.0375017872152676, -1.1577464368965131, -1.5056327880797116, -1.5064026167930198, -1.0048538489695498, -0.7021565996138799, -0.2895347302068554, -0.1039928018155829],
				  [0.5910430676424782, 1.0064369818356056, -1.2668176821830330, 0.6160250511673788, 1.0375017872152676, -1.1577464368965131, -1.5056327880797116, -1.5064026167930198, -1.0048538489695498, -0.7021565996138799, -0.2895347302068554, -0.1039928018155829],
				  [0.5910430676424782, 1.0064369818356056, -1.2668176821830330, 0.6160250511673788, 1.0375017872152676, -1.1577464368965131, -1.5056327880797116, -1.5064026167930198, -1.0048538489695498, -0.7021565996138799, -0.2895347302068554, -0.1039928018155829]]
        des_gt = [[1, 0, 0, 1]]
        des_rp = None
        des_ro = None
        np.testing.assert_array_almost_equal(la, des_la, 14)
        np.testing.assert_array_almost_equal(av, des_av, 14)
        np.testing.assert_array_almost_equal(p, des_p, 14)
        np.testing.assert_array_almost_equal(v, des_v, 14)
        np.testing.assert_equal(t, des_t)
        np.testing.assert_array_almost_equal(fp, des_fp, 14)
        np.testing.assert_array_almost_equal(fv, des_fv, 14)
        np.testing.assert_array_almost_equal(gt, des_gt, 14)
        np.testing.assert_equal(r_p, des_rp)
        np.testing.assert_equal(r_o, des_ro)

class TestQuadSDKDatasets(unittest.TestCase):
    """
    Test that the QuadSDK dataset class successfully
    processes and reads info from the dataset.
    """

    def setUp(self):
        # Get the paths to the URDF file and the dataset
        self.path_to_a1_urdf = Path(
            Path('.').parent, 'urdf_files', 'A1', 'a1.urdf').absolute()
        self.path_to_normal_sequence = Path(
            Path('.').parent, 'datasets', 'QuadSDK-A1Speed1.0').absolute()

        # Set up the QuadSDK dataset
        self.dataset_hgnn_1 = QuadSDKDataset_A1Speed1_0(self.path_to_normal_sequence,
            self.path_to_a1_urdf, 'package://a1_description/',
                            'unitree_ros/robots/a1_description', 'heterogeneous_gnn', 1)

    def test_process_and_load_data_at_ros_seq(self):
        """
        Make sure the data is processed and loaded properly from the file.
        """

        la, av, p, v, t, fp, fv, gt, r_p, r_o = self.dataset_hgnn_1.load_data_at_dataset_seq(10000)
        des_la = [[-0.06452160178213015, -0.366493877667443, 9.715652148737323]]
        des_av = [[-0.0017398309484803294, -0.011335050676391335, 1.2815129213608234]]
        des_p = [[-0.16056788963386381,  0.6448773402529877, 1.1609664103261004, -0.1931352599922853,
                 0.4076711540455795,  0.9424138768126973,  0.11254642823264671, 0.5695073200020913,
                 0.9825683053175158, 0.22800622574234453, 0.4399285706508147, 1.1786520769378077]]
        des_v = [[-1.015061543404335, 0.459564643757568, 0.15804277355899754, 1.9489188274516005,
                  1.3299772937985548, 3.644698146547278, -1.2845189574751656, 2.2337115054710917,
                  3.4964483387811476, 1.020374615076573, -0.34825271015763287, 0.3185087654826033]]
        des_t = [[-3.658930090797254, -3.858403899440098, 12.475195605359755, 0.6111713969715354,
                 -0.11888726638996594, -0.24871924601280232, -0.6701594400127251,-0.48841280756095506,
                 -0.1350813273560049, 3.6351217639084576, 4.456326408036115, 7.829759255207876]]
        des_fp = None
        des_fv = None
        des_gt = [[64.74924447333427, 0, 0, 64.98097097053076]]
        des_rp = [[-0.9077062285177977, 0.400808137988427, 0.25139755534263925]]
        des_ro = [[0.00807844275531408, 0.006594002480070749, -0.6310441334749343, -0.7756768396057802]]
        np.testing.assert_array_equal(la, des_la)
        np.testing.assert_array_equal(av, des_av)
        np.testing.assert_array_equal(p, des_p)
        np.testing.assert_array_equal(v, des_v)
        np.testing.assert_array_equal(t, des_t)
        np.testing.assert_equal(fp, des_fp)
        np.testing.assert_equal(fv, des_fv)
        np.testing.assert_array_equal(gt, des_gt)
        np.testing.assert_array_almost_equal(r_p, des_rp, 16)
        np.testing.assert_array_almost_equal(r_o, des_ro, 16)

class TestLinTzuYaunDatasets(unittest.TestCase):
    """
    Test that the class for the MIT Mini Cheetah Contact Dataset
    found at https://github.com/UMich-CURLY/deep-contact-estimator
    properly loads and returns data.
    """

    def setUp(self):
        # Get the paths to the URDF file and the dataset
        self.path_to_mc_urdf = Path('urdf_files', 'MiniCheetah', 'miniCheetah.urdf').absolute()
        self.path_to_crc_seq = Path(Path('.').parent, 'datasets', 'LinTzuYaun-CRC').absolute()

        # Setup the datasets for testing
        self.dataset_mlp_1 = LinTzuYaunDataset_concrete_right_circle(
            self.path_to_crc_seq, self.path_to_mc_urdf, 'package://yobotics_description/',
            'mini-cheetah-gazebo-urdf/yobo_model/yobotics_description', 'mlp', 1)

    def test_process_and_load_data_at_ros_seq(self):
        """
        Make sure the data is properly processed and loaded from the dataset.
        """

        la, av, p, v, t, fp, fv, gt, r_p, r_o = self.dataset_mlp_1.load_data_at_dataset_seq(30000)
        des_la = [[0.146148949861527,	0.0223360173404217,	9.58438873291016]]
        des_av = [[0.00771053740754724,	0.0315540358424187,	0.161459520459175]]
        des_p = [[0.0581750869750977,	-0.498899102210999,	1.01744437217712,	
                 0.000190734863281250,	-0.625548958778381,	1.45497155189514,	
                 0.00858306884765625, -0.650104880332947,	1.49907255172730,	
                 0.0394830703735352,	-0.420838713645935,	0.846460103988648]]
        des_v = [[0.0793685913085938,	-1.38095092773438,	0.0102071752771735,	
                 0.142860412597656,	8.93650817871094,	-9.21489906311035,	
                 -1.60317611694336,	9.79365158081055,	-9.19449234008789,	
                 0.301589965820313,	-1.47618865966797,	-0.418396085500717]]
        des_t = None
        des_fp = [[-0.00335261225700378,	-0.0453704893589020,	-0.356130868196487,	
                  0.0214422345161438,	0.0660574287176132,	-0.301095396280289,	
                  0.0198653340339661,	-0.0634637400507927,	-0.295771747827530,	
                  -0.00486903637647629,	0.0804890319705010,	-0.365474313497543]]
        des_fv = [[-0.485595673322678,	0.0279390625655651,	0.00200577825307846,	
                  1.47740077972412,	0.0432308465242386,	-1.12423670291901,	
                  1.70655310153961,	-0.464293301105499,	-1.04942762851715,	
                  -0.618086814880371,	0.111269362270832, -0.00220335274934769]]
        des_gt = [[1, 0, 0, 1]]
        des_rp = None
        des_ro = None
        np.testing.assert_array_almost_equal(la, des_la, 14)
        np.testing.assert_array_almost_equal(av, des_av, 14)
        np.testing.assert_array_almost_equal(p, des_p, 14)
        np.testing.assert_array_almost_equal(v, des_v, 14)
        np.testing.assert_equal(t, des_t)
        np.testing.assert_array_almost_equal(fp, des_fp, 14)
        np.testing.assert_array_almost_equal(fv, des_fv, 14)
        np.testing.assert_array_almost_equal(gt, des_gt, 14)
        np.testing.assert_equal(r_p, des_rp)
        np.testing.assert_equal(r_o, des_ro)

if __name__ == "__main__":
    unittest.main()
