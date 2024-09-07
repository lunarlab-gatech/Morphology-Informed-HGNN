import unittest
from pathlib import Path
from mi_hgnn import FlexibleDataset, QuadSDKDataset_A1Speed1_0_DEPRECATED, QuadSDKDataset, QuadSDKDataset_A1Speed0_5_DEPRECATED
from mi_hgnn.datasets_py.quadSDKDataset import *
from mi_hgnn.datasets_py.LinTzuYaunDataset import LinTzuYaunDataset, LinTzuYaunDataset_concrete_right_circle
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
        self.path_to_go2_quad_urdf = Path('urdf_files', 'Go2-Quad', 'go2.urdf').absolute()
        
        # Set up the paths to the datasets
        self.path_to_quad_seq = Path(
            Path('.').parent, 'datasets', 'QuadSDK-A1Speed1.0').absolute()
        self.path_to_lin_seq = Path(Path('.').parent, 'datasets', 'LinTzuYaun-CRC').absolute()
        self.path_to_go2_seq = Path(Path('.').parent, 'datasets', 'QuadSDK-Go2-Flat-0.5-Mu50').absolute()

        # Set up the QuadSDK datasets
        self.quad_dataset_hgnn_1 = QuadSDKDataset_A1Speed1_0_DEPRECATED(self.path_to_quad_seq,
            self.path_to_a1_urdf, 'package://a1_description/',
                            'unitree_ros/robots/a1_description', 'heterogeneous_gnn', 1)
        self.quad_dataset_mlp_1 = QuadSDKDataset_A1Speed1_0_DEPRECATED(self.path_to_quad_seq,
            self.path_to_a1_urdf, 'package://a1_description/',
                            'unitree_ros/robots/a1_description', 'mlp', 1)
        self.quad_dataset_hgnn_3 = QuadSDKDataset_A1Speed1_0_DEPRECATED(self.path_to_quad_seq,
            self.path_to_a1_urdf, 'package://a1_description/',
                            'unitree_ros/robots/a1_description', 'heterogeneous_gnn', 3)
        self.quad_dataset_mlp_3 = QuadSDKDataset_A1Speed1_0_DEPRECATED(self.path_to_quad_seq,
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
        self.quad_dataset_hgnn_6_norm = QuadSDKDataset_A1Speed1_0_DEPRECATED(self.path_to_quad_seq,
            self.path_to_a1_urdf, 'package://a1_description/',
            'unitree_ros/robots/a1_description', 'heterogeneous_gnn', 
            6, normalize=True)
        self.lin_dataset_hgnn_10_norm = LinTzuYaunDataset_concrete_right_circle(
            self.path_to_lin_seq, self.path_to_mc_urdf, 'package://yobotics_description/',
            'mini-cheetah-gazebo-urdf/yobo_model/yobotics_description', 'heterogeneous_gnn', 
            10, normalize=True)
        
        # Set up dataset to test 'dynamics' model
        self.dataset_go2_dynamics = QuadSDKDataset_Go2_Flat_Speed0_5_Mu_50(
            self.path_to_go2_seq, self.path_to_go2_quad_urdf, 'package://go2_description/', 
            '', 'dynamics', 1, normalize=False)

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
        dataset_slow = QuadSDKDataset_A1Speed0_5_DEPRECATED(path_to_slow_sequence,
                            self.path_to_a1_urdf, 'package://a1_description/',
                            'unitree_ros/robots/a1_description', 'heterogeneous_gnn', 1)
        
        # Try to create a normal sequence, pointing to the slow dataset directory
        with self.assertRaises(ValueError):
            dataset = QuadSDKDataset_A1Speed0_5_DEPRECATED(self.path_to_quad_seq,
                                 self.path_to_a1_urdf, 'package://a1_description/',
                                'unitree_ros/robots/a1_description', 'heterogeneous_gnn', 1)
            
        # Try to create a slow sequence, pointing to the normal dataset directory
        with self.assertRaises(ValueError) as e:
            dataset = QuadSDKDataset_A1Speed1_0_DEPRECATED(path_to_slow_sequence,
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
            
        # ============== Test dynamics model can only be created with history_length = 1 ==============
        with self.assertRaises(ValueError):
            QuadSDKDataset_Go2_Flat_Speed0_5_Mu_50(self.path_to_go2_seq, self.path_to_go2_quad_urdf, 
                'package://go2_description/', 'unitree_ros/robots/go2_description', 'dynamics', 5, normalize=False)
        
        
    def test_load_data_sorted(self):
        """
        Test that the sorting function matches the order provided by the child class.
        """
        # ====================== Test QuadSDK dataset ======================
        la, av, p, v, t, fp, fv, gt, r_p, r_o, ts = self.quad_dataset_hgnn_1.load_data_sorted(10000)
        des_la = [[-0.06452160178213015, -0.366493877667443, 9.715652148737323]]
        des_av = [[-0.0017398309484803294, -0.011335050676391335, 1.2815129213608234]]
        des_p = [[0.11254642823264671, 0.5695073200020913, 0.9825683053175158,
                 -0.16056788963386381,  0.6448773402529877, 1.1609664103261004,
                 0.22800622574234453, 0.4399285706508147, 1.1786520769378077,
                 -0.1931352599922853,  0.4076711540455795,  0.9424138768126973]]
        des_v = [[-1.2845189574751656, 2.2337115054710917, 3.4964483387811476,
                  -1.015061543404335, 0.459564643757568, 0.15804277355899754,
                  1.020374615076573, -0.34825271015763287, 0.3185087654826033,
                  1.9489188274516005, 1.3299772937985548, 3.644698146547278]]
        des_t = [[-0.6701594400127251,-0.48841280756095506, -0.1350813273560049,
                  -3.658930090797254, -3.858403899440098, 12.475195605359755,
                  3.6351217639084576, 4.456326408036115, 7.829759255207876,
                 0.6111713969715354, -0.11888726638996594, -0.24871924601280232]]
        des_fp = None
        des_fv = None
        des_gt = [0, 64.74924447333427, 64.98097097053076, 0]
        des_rp = [[-0.9077062285177977, 0.400808137988427, 0.25139755534263925]]
        des_ro = [[0.00807844275531408, 0.006594002480070749, -0.6310441334749343, -0.7756768396057802]]
        des_ts = [[27.374, 27.375, 27.374]]
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
        np.testing.assert_equal(ts, des_ts)

        # ===================== Test LinTzuYaun dataset ====================
        la, av, p, v, t, fp, fv, gt, r_p, r_o, ts = self.lin_dataset_mlp_1.load_data_sorted(30000)
        des_la = [[0.146148949861527,	0.0223360173404217,	9.58438873291016]]
        des_av = [[0.00771053740754724,	0.0315540358424187,	0.161459520459175]]
        des_p = [[0.0394830703735352,	-0.420838713645935,	0.846460103988648,	
                 0.000190734863281250,	-0.625548958778381,	1.45497155189514,	
                 0.00858306884765625, -0.650104880332947,	1.49907255172730,	
                 0.0581750869750977,	-0.498899102210999,	1.01744437217712]]
        des_v = [[0.301589965820313,	-1.47618865966797,	-0.418396085500717,
                 0.142860412597656,	8.93650817871094,	-9.21489906311035,	
                 -1.60317611694336,	9.79365158081055,	-9.19449234008789,	
                 0.0793685913085938,	-1.38095092773438,	0.0102071752771735]]
        des_t = None
        des_fp = [[-0.00486903637647629,	0.0804890319705010,	-0.365474313497543,
                  0.0214422345161438,	0.0660574287176132,	-0.301095396280289,	
                  0.0198653340339661,	-0.0634637400507927,	-0.295771747827530,	
                  -0.00335261225700378,	-0.0453704893589020,	-0.356130868196487]]
        des_fv = [[-0.618086814880371,	0.111269362270832, -0.00220335274934769,
                  1.47740077972412,	0.0432308465242386,	-1.12423670291901,	
                  1.70655310153961,	-0.464293301105499,	-1.04942762851715,	
                  -0.485595673322678,	0.0279390625655651,	0.00200577825307846]]
        des_gt = [1, 0, 0, 1]
        des_rp = None
        des_ro = None
        des_ts = None
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
        np.testing.assert_equal(ts, des_ts)

    def test_get_helper_mlp(self):
        # ====================== Test QuadSDK dataset ======================
        x, y = self.quad_dataset_mlp_1.get_helper_mlp(10000)

        des_x = np.array([-0.06452160178213015, -0.366493877667443, 9.715652148737323,
                 -0.0017398309484803294, -0.011335050676391335, 1.2815129213608234,
                  0.11254642823264671, 0.5695073200020913, 0.9825683053175158,
                 -0.16056788963386381,  0.6448773402529877, 1.1609664103261004,
                  0.22800622574234453, 0.4399285706508147, 1.1786520769378077,
                 -0.1931352599922853,  0.4076711540455795,  0.9424138768126973,
                 -1.2845189574751656, 2.2337115054710917, 3.4964483387811476,
                 -1.015061543404335, 0.459564643757568, 0.15804277355899754,
                  1.020374615076573, -0.34825271015763287, 0.3185087654826033,
                  1.9489188274516005, 1.3299772937985548, 3.644698146547278,
                 -0.6701594400127251,-0.48841280756095506, -0.1350813273560049,
                 -3.658930090797254, -3.858403899440098, 12.475195605359755,
                  3.6351217639084576, 4.456326408036115, 7.829759255207876,
                  0.6111713969715354, -0.11888726638996594, -0.24871924601280232], dtype=np.float64)
        des_y = np.array([0, 64.74924447333427, 64.98097097053076, 0], dtype=np.float64)

        np.testing.assert_array_equal(y, des_y)
        np.testing.assert_array_equal(x, des_x)

        # ===================== Test LinTzuYaun dataset ====================
        x, y = self.lin_dataset_mlp_1.get_helper_mlp(30000)

        des_x = np.array([0.146148949861527,	0.0223360173404217,	9.58438873291016,
                 0.00771053740754724,	0.0315540358424187,	0.161459520459175,
                 0.0394830703735352,	-0.420838713645935,	0.846460103988648,	
                 0.000190734863281250,	-0.625548958778381,	1.45497155189514,	
                 0.00858306884765625, -0.650104880332947,	1.49907255172730,	
                 0.0581750869750977,	-0.498899102210999,	1.01744437217712,
                 0.301589965820313,	-1.47618865966797,	-0.418396085500717,
                 0.142860412597656,	8.93650817871094,	-9.21489906311035,	
                -1.60317611694336,	9.79365158081055,	-9.19449234008789,	
                 0.0793685913085938,	-1.38095092773438,	0.0102071752771735,
                -0.00486903637647629,	0.0804890319705010,	-0.365474313497543,
                 0.0214422345161438,	0.0660574287176132,	-0.301095396280289,	
                 0.0198653340339661,	-0.0634637400507927,	-0.295771747827530,	
                -0.00335261225700378,	-0.0453704893589020,	-0.356130868196487,
                -0.618086814880371,	0.111269362270832, -0.00220335274934769,
                 1.47740077972412,	0.0432308465242386,	-1.12423670291901,	
                 1.70655310153961,	-0.464293301105499,	-1.04942762851715,	
                -0.485595673322678,	0.0279390625655651,	0.00200577825307846], dtype=np.float64)
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

    def test_get_helper_dynamics(self):
        q, vel, acc, tau, labels = self.dataset_go2_dynamics.get(0)

        # Set the desired values
        des_q = np.array([0.11106036, 0.0219779, 0.26230157, #DONE
                          -0.01051185, -0.02242382, 0.0031927, 0.99968819,  
                          -0.00328921,  0.51851414,  1.03396776, 
                           0.01678816 , 0.64268708,  1.26193817,
                           0.00985092,  0.67495145,  1.34233534,
                          -0.0077084 ,  0.48193389,  0.98210022], dtype=np.double)
        des_vel = np.array([ 0.28714938,  0.09367028, -0.01043991,    # DONE
                             -0.17603125,  0.01065967,  0.26195285,
                              0.34195181,  2.95817075, -0.0123834,  
                              0.00966363, -1.03752528,  0.40627908,
                             -0.3197401,  -1.02283784, -0.14568963,
                              0.85652331,  3.89292041,  0.62152583,  
                             ], dtype=np.double)
        des_acc = np.array([0.34369415,  0.17348376, 10.62882022, # DONE
                             5.59120484, -0.10766368, -0.14405578,
                             0.34615535,  48.76842517, 103.9891471,  
                             -8.56842182, 5.97236389,   6.49158482,
                             -8.97923037,   5.7985695,    7.36054217,  
                             -0.28729473,  44.51742379, 106.12421164,      
                             ], dtype=np.double)
        des_tau = np.array([0, 0, 0,  # DONE
                             0, 0, 0,
                             -0.34050237, -0.08218517,  0.19674661,  
                             4.39419366, -0.21540196, 11.43988289,
                             -5.75678065, -0.93363904, 10.04449209,  
                             0.72560437, -0.21097773,  0.15044776,
                             ], dtype=np.double)
        des_labels = np.array([0, 71.78180083, 63.22432284, 0], dtype=np.double)

        # Test that the values are correct
        np.testing.assert_array_almost_equal(q.numpy(), des_q, 8)
        np.testing.assert_array_almost_equal(vel.numpy(), des_vel, 8)
        np.testing.assert_array_almost_equal(acc.numpy(), des_acc, 6)
        np.testing.assert_array_almost_equal(tau.numpy(), des_tau, 8)
        np.testing.assert_array_almost_equal(labels.numpy(), des_labels, 8)

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

        # Test that the dynamics model has a bound 2 less than the
        # number of dataset entries
        with self.assertRaises(IndexError):
            data = self.dataset_go2_dynamics.get(-1)
        with self.assertRaises(IndexError):
            data = self.dataset_go2_dynamics.get(12041)
        data = self.dataset_go2_dynamics.get(0)
        data = self.dataset_go2_dynamics.get(12040)

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
            
            la, av, p, v, t, fp, fv, gt, r_p, r_o, ts = hgnn_1_datasets[i].load_data_sorted(10000)
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
        la, av, p, v, t, fp, fv, gt, r_p, r_o, ts = self.quad_dataset_hgnn_6_norm.load_data_sorted(10000)

        des_la = [[-0.5362300252378239, -1.6632797193920590, -1.5898042183134058],
				  [0.0423700008727095, -0.2257869376580193, -0.3294529164335535],
				  [0.0423700008727095, -0.2257869376580193, -0.3294529164335535],
				  [-0.0272254120334773, 0.0614106314680039, 0.3784460702629509],
				  [-1.2761523397622021, 1.1253516629229292, 1.3613238439024378],
				  [1.7548677752880986, 0.9280913003171616, 0.5089401370151242]]
        des_av = [[0.3922097071448895, -1.9912456309175439, -1.6041605549730955],
				  [-0.9026040953610488, 0.1920045172187222, -0.2748544676177472],
				  [-0.9026040953610488, 0.1920045172187222, -0.2748544676177472],
				  [-0.7582570552051391, 0.7857144431190286, 0.0388281714139248],
				  [0.7083651482760132, 0.3541357390230436, 0.9208249386706086],
				  [1.4628903905063348, 0.4673864143380300, 1.1942163801239956]]
        des_p =  [[1.3411660106298753, -1.3464441876713720, -1.3329276164891533, 1.3251319388825571, -1.3292953444351709, 1.3563194175810893, -1.3399555835689183, 1.3392325382228460, 1.3353606785305967, -1.3458357606384972, -1.3105046876188471, -1.3327600834631594],
				  [0.7989948511378354, -0.8004879787140102, -0.8034252146278538, 0.8016165704870016, -0.8024331988028045, 0.8022555036688123, -0.8035576420140235, 0.7999639528511129, 0.7762335698833166, -0.8015405731840229, -0.8075165530500278, -0.8029836562059367],
				  [0.2622460474785945, -0.2590229953949031, -0.2699089968205644, 0.2753282103905000, -0.2720028085194331, 0.2521407814643208, -0.2650510327335250, 0.2650531146443696, 0.2656168693488439, -0.2604470625273131, -0.2872772764940732, -0.2698329209686913],
				  [-0.2684005798638849, 0.2764196967898784, 0.2658743310353441, -0.2552066579712369, 0.2604971933717123, -0.2894640820782961, 0.2735157756310986, -0.2681152978110906, -0.2341061531035037, 0.2775733289258568, 0.2477837612713722, 0.2650490395306883],
				  [-0.7976728914440070, 0.8046277353998353, 0.8021988335881026, -0.7941729446190279, 0.7972959131392616, -0.8161211101489749, 0.8081642513263279, -0.8008219243499484, -0.7613381642897331, 0.8078348135943265, 0.7966807098264718, 0.8010262195325177],
				  [-1.3363334379384193, 1.3249077295905987, 1.3381886633141249, -1.3526971171697941, 1.3459382452462334, -1.3051305104869513, 1.3268842313590543, -1.3353123835572895, -1.3817668003729298, 1.3224152538296576, 1.3608340460651245, 1.3395014015745652]]
        des_v =  [[-1.5434440927124917, 1.1997091543290412, -1.6636050201412875, 0.6755515856359472, -0.9091400983508593, -0.7956179650646124, -0.0122222758847872, -1.7130430801273044, -1.3509086925789371, 0.8157856644249428, -1.3655521804259694, -1.4021887223182439],
				  [-0.3378426294346330, 0.7826872385979762, -0.7742568976523585, 0.7678512617947539, -0.8622849372938384, -0.8467797442950677, 0.6030870998585781, -0.6671475072399285, -0.1548234510322314, 0.7570222876049989, -0.8184315138209003, -0.8961639470728695],
				  [0.2232026398243473, 0.4089795728204378, 0.2865436738062712, 0.5379423154263993, -0.3909533313924068, -0.5258103600852684, 0.6834587531357984, 0.3066748880030281, 0.7593058641579262, 0.4589785782876333, -0.1992585062162287, 0.0423590598220767],
				  [1.0644798416901866, -0.1444347058132701, 0.7262730385930941, 0.2563519144747153, -0.0596892679942115, -0.0758937147025660, 0.6285537750737797, 0.6810366845742841, 1.0388841464697509, 0.2433269329503818, 0.2901287243831323, 0.3293674576395288],
				  [1.0532817768073961, -0.7802502608673852, 0.7757201940447753, -0.3788174071672958, 0.4767115945407807, 0.4702367625833525, 0.0393866722499664, 0.9119224381390747, 0.7059693814995577, -0.4654440034437909, 0.7717873266845418, 0.5996266793123807],
				  [-0.4596775361748052, -1.4666909990668091, 0.6493250113494371, -1.8588796701644894, 1.7453560404904871, 1.7738650215641547, -1.9422640244333413, 0.4805565766508563, -0.9984272485161201, -1.8096694598241658, 1.3213261493954216, 1.3269994726171432]]
        des_t =  [[-0.4450888677280738, 0.8342683245847919, 0.8724131545018894, 0.6070862499216861, 1.2368789594030656, -1.1882549109956477, 1.1546556675456252, -1.4533214173843505, 1.0246572695927316, -0.6051082553693820, 0.8991887934850253, 0.4346911998392983],
				  [-0.3723978523667422, 1.2539079576422159, 1.3072191011585699, 0.4331492429177987, 0.9992594011349417, -0.6976114163880145, 0.7975706261921119, -0.8077307173252027, 0.8989505370730972, -0.7350301589189259, 1.4322130031735478, 1.5153849947456275],
				  [1.4072471903113419, 0.2766554342964559, 0.2225920671658350, 0.3882887341670302, -0.2243124086011362, -0.2735366520870312, 0.3149725399596286, -0.0933332335012110, 0.3538973903559970, 0.6618893260166033, -0.2643698397474668, -0.6728053368920602],
				  [-0.8524164667010604, -0.4923671985869356, -0.4544701090098361, 0.6852826820416711, -0.0502935684188838, -0.1283714588583099, 0.0263215255680864, 0.2937590539414849, 0.0692342887903731, -0.4204548692029078, -0.4784159228251703, -0.5158614590193742],
				  [-0.8489176765225551, -0.3619127384397345, -0.5396479264344930, -0.1654558400344384, -0.4832921001935127, 0.7005300796033610, -0.7590534329665927, 0.9372216207755134, -0.8139705532946192, -0.6496308599111635, -0.2559764426609154, 0.4870509333260056],
				  [1.1115736730070749, -1.5105517794968020, -1.4081062873819652, -1.9483510690137440, -1.4782402833244745, 1.5872443587256342, -1.5344669262988595, 1.1234046934937503, -1.5327689325175256, 1.7483348173857820, -1.3326395914250233, -1.2484603319994980]]
        des_fp = None
        des_fv = None
        des_gt = [0, 66.058643452423740, 65.599211397640180, 0] # Not normalized
        des_rp = [[1.3367302859481276, -1.3404977880696314, -1.3404771033469411],
				  [0.7972891276500045, -0.8007337701560415, -0.8002318101735841],
				  [0.2655899408453240, -0.2638863991129750, -0.2635796630011911],
				  [-0.2616923371464652, 0.2703377880188639, 0.2697412648386170],
				  [-0.7922667061527424, 0.8023152154400157, 0.8004689671890513],
				  [-1.3456503111442484, 1.3324649538797682, 1.3340783444954187]]
        des_ro = [[-1.2885895503537950, -0.3468726736045049, 1.3342278298107564, -1.3320142312727166],
				  [-0.4038681453011558, 0.2194056555128877, 0.8022566955585412, -0.8027296043221029],
				  [0.4698314619749407, 0.6738262671479421, 0.2689298724716307, -0.2707074699423853],
				  [1.0870010249865283, 0.8925717448025465, -0.2656824006915559, 0.2639630143328630],
				  [1.0072053769211737, 0.4087085025436538, -0.8014601785627550, 0.8010978846144309],
				  [-0.8715801682276921, -1.8476394964022487, -1.3382718185866767, 1.3403904065904897]]
        des_ts = [[27.374, 27.375, 27.374],
                    [27.376, 27.377, 27.377],
                    [27.378, 27.379, 27.378],
                    [27.38,  27.381, 27.38 ],
                    [27.382, 27.383, 27.382],
                    [27.384, 27.385, 27.384]]
        np.testing.assert_array_almost_equal(la, des_la, 13)
        np.testing.assert_array_almost_equal(av, des_av, 13)
        np.testing.assert_array_almost_equal(p, des_p, 13)
        np.testing.assert_array_almost_equal(v, des_v, 13)
        np.testing.assert_array_almost_equal(t, des_t, 13)
        np.testing.assert_array_equal(fp, des_fp)
        np.testing.assert_array_equal(fv, des_fv)
        np.testing.assert_array_almost_equal(gt, des_gt, 13)
        np.testing.assert_array_almost_equal(r_p, des_rp, 11)
        np.testing.assert_array_almost_equal(r_o, des_ro, 12)
        np.testing.assert_array_almost_equal(ts, des_ts)

        # ===================== Test LinTzuYaun dataset ====================
        la, av, p, v, t, fp, fv, gt, r_p, r_o, ts = self.lin_dataset_hgnn_10_norm.load_data_sorted(30000)
        des_la = [[1.4491376746189437, -1.4491376746189439, -1.4491376746189455],
				  [1.4491376746189437, -1.4491376746189439, -1.4491376746189455],
				  [1.4491376746189437, -1.4491376746189439, -1.4491376746189455],
				  [-0.6210590034081187, 0.6210590034081187, 0.6210590034081173],
				  [-0.6210590034081187, 0.6210590034081187, 0.6210590034081173],
				  [-0.6210590034081187, 0.6210590034081187, 0.6210590034081173],
				  [-0.6210590034081187, 0.6210590034081187, 0.6210590034081173],
				  [-0.6210590034081187, 0.6210590034081187, 0.6210590034081173],
				  [-0.6210590034081187, 0.6210590034081187, 0.6210590034081173],
				  [-0.6210590034081187, 0.6210590034081187, 0.6210590034081173]]
        des_av = [[-1.4491376746189435, 1.4491376746189437, -1.4491376746189486],
				  [-1.4491376746189435, 1.4491376746189437, -1.4491376746189486],
				  [-1.4491376746189435, 1.4491376746189437, -1.4491376746189486],
				  [0.6210590034081187, -0.6210590034081188, 0.6210590034081144],
				  [0.6210590034081187, -0.6210590034081188, 0.6210590034081144],
				  [0.6210590034081187, -0.6210590034081188, 0.6210590034081144],
				  [0.6210590034081187, -0.6210590034081188, 0.6210590034081144],
				  [0.6210590034081187, -0.6210590034081188, 0.6210590034081144],
				  [0.6210590034081187, -0.6210590034081188, 0.6210590034081144],
				  [0.6210590034081187, -0.6210590034081188, 0.6210590034081144]]
        des_p = [[-1.6431676725154987, 1.3954241717130242, 1.4907544284266612, -0.6210590034081188, -1.4907880397936644, 1.4833054412157543, 1.5527892963392784, -1.5129981064756799, 1.4764001117023364, 0.0000000000000000, 1.5636397280635477, 1.8973665961005983],
				  [-1.6431676725154987, 1.3954241717130242, 1.4907544284266612, -0.6210590034081188, -1.4907880397936644, 1.4833054412157543, 1.5527892963392784, -1.5129981064756799, 1.4764001117023364, 0.0000000000000000, 1.5636397280635477, 1.8973665961005983],
				  [0.0000000000000000, 0.2078291319572589, 0.1355530739847286, -0.6210590034081188, -0.1355261854357877, 0.1414854895601932, 0.0839345565588799, -0.1175144160369460, 0.1469382537512688, 0.0000000000000000, 0.0744590346696927, -0.4743416490256865],
				  [0.0000000000000000, 0.2078291319572589, 0.1355530739847286, -0.6210590034081188, -0.1355261854357877, 0.1414854895601932, 0.0839345565588799, -0.1175144160369460, 0.1469382537512688, 0.0000000000000000, 0.0744590346696927, -0.4743416490256865],
				  [0.0000000000000000, 0.2078291319572589, 0.1355530739847286, -0.6210590034081188, -0.1355261854357877, 0.1414854895601932, 0.0839345565588799, -0.1175144160369460, 0.1469382537512688, 0.0000000000000000, 0.0744590346696927, -0.4743416490256865],
				  [0.0000000000000000, 0.2078291319572589, 0.1355530739847286, -0.6210590034081188, -0.1355261854357877, 0.1414854895601932, 0.0839345565588799, -0.1175144160369460, 0.1469382537512688, 0.0000000000000000, 0.0744590346696927, -0.4743416490256865],
				  [0.0000000000000000, 0.2078291319572589, 0.1355530739847286, -0.6210590034081188, -0.1355261854357877, 0.1414854895601932, 0.0839345565588799, -0.1175144160369460, 0.1469382537512688, 0.0000000000000000, 0.0744590346696927, -0.4743416490256865],
				  [1.0954451150103324, -1.2766646677374476, -1.2197580755924582, 1.4491376746189439, 1.2197356689220891, -1.2246794434108101, -1.1750837918243189, 1.2045227643786967, -1.2291638307203390, 0.0000000000000000, -1.1665248764918530, -0.4743416490256865],
				  [1.0954451150103324, -1.2766646677374476, -1.2197580755924582, 1.4491376746189439, 1.2197356689220891, -1.2246794434108101, -1.1750837918243189, 1.2045227643786967, -1.2291638307203390, 0.0000000000000000, -1.1665248764918530, -0.4743416490256865],
				  [1.0954451150103324, -1.2766646677374476, -1.2197580755924582, 1.4491376746189439, 1.2197356689220891, -1.2246794434108101, -1.1750837918243189, 1.2045227643786967, -1.2291638307203390, 0.0000000000000000, -1.1665248764918530, -0.4743416490256865]]
        des_v = [[-1.6431676725154984, -1.3671981455629505, 1.5790637348111864, 0.9486832980505138, -0.0729756383115739, 1.4342795371518564, 0.6210590034081203, -0.4936055194583740, 1.6906341359604951, 0.9486832980505139, 1.2576675135944444, 1.8180368816854224],
				  [-1.6431676725154984, -1.3671981455629505, 1.5790637348111864, 0.9486832980505138, -0.0729756383115739, 1.4342795371518564, 0.6210590034081203, -0.4936055194583740, 1.6906341359604951, 0.9486832980505139, 1.2576675135944444, 1.8180368816854224],
				  [0.8215838362577492, 0.9114654303752995, -0.8502651903029340, -0.9486832980505138, -0.8027320214273539, 0.1792803869262010, 0.6210590034081203, 0.9166959647084314, -0.0497251740771810, -0.9486832980505139, -0.9295803361350234, -0.6895924092203737],
				  [0.8215838362577492, 0.9114654303752995, -0.8502651903029340, -0.9486832980505138, -0.8027320214273539, 0.1792803869262010, 0.6210590034081203, 0.9166959647084314, -0.0497251740771810, -0.9486832980505139, -0.9295803361350234, -0.6895924092203737],
				  [0.8215838362577492, 0.9114654303752995, -0.8502651903029340, -0.9486832980505138, -0.8027320214273539, 0.1792803869262010, 0.6210590034081203, 0.9166959647084314, -0.0497251740771810, -0.9486832980505139, -0.9295803361350234, -0.6895924092203737],
				  [0.8215838362577492, 0.9114654303752995, -0.8502651903029340, -0.9486832980505138, -0.8027320214273539, 0.1792803869262010, 0.6210590034081203, 0.9166959647084314, -0.0497251740771810, -0.9486832980505139, -0.9295803361350234, -0.6895924092203737],
				  [0.8215838362577492, 0.9114654303752995, -0.8502651903029340, -0.9486832980505138, -0.8027320214273539, 0.1792803869262010, 0.6210590034081203, 0.9166959647084314, -0.0497251740771810, -0.9486832980505139, -0.9295803361350234, -0.6895924092203737],
				  [-0.2738612787525831, -0.6076436202502005, 0.3643994939640978, 0.9486832980505138, 1.3865371279199861, -1.2549870029782342, -1.4491376746189424, -1.1987562615417766, -1.0442141338450321, 0.9486832980505139, 0.7108555511620774, -0.0627039057563253],
				  [-0.2738612787525831, -0.6076436202502005, 0.3643994939640978, 0.9486832980505138, 1.3865371279199861, -1.2549870029782342, -1.4491376746189424, -1.1987562615417766, -1.0442141338450321, 0.9486832980505139, 0.7108555511620774, -0.0627039057563253],
				  [-0.2738612787525831, -0.6076436202502005, 0.3643994939640978, 0.9486832980505138, 1.3865371279199861, -1.2549870029782342, -1.4491376746189424, -1.1987562615417766, -1.0442141338450321, 0.9486832980505139, 0.7108555511620774, -0.0627039057563253]]
        des_fp = [[1.4120404060924265, -1.6425152893883035, -1.7987241618517256, -1.4935355815039442, -0.6250284357800648, 1.4951317978279843, -1.5364377865823451, 1.5493517906132577, 1.4705132221241244, 1.5825504151799377, -1.4837951673595022, 1.4842824997329462],
				  [1.4120404060924265, -1.6425152893883035, -1.7987241618517256, -1.4935355815039442, -0.6250284357800648, 1.4951317978279843, -1.5364377865823451, 1.5493517906132577, 1.4705132221241244, 1.5825504151799377, -1.4837951673595022, 1.4842824997329462],
				  [0.1957627154676198, -0.0006519950679582, 0.1882218934706221, -0.1333245020495574, -0.6194690977235712, 0.1320420401250590, -0.0979484326519782, 0.0869066547652635, 0.1515520936725435, 0.0575871300734228, 0.8829946701584727, -0.8828516073209223],
				  [0.1957627154676198, -0.0006519950679582, 0.1882218934706221, -0.1333245020495574, -0.6194690977235712, 0.1320420401250590, -0.0979484326519782, 0.0869066547652635, 0.1515520936725435, 0.0575871300734228, 0.8829946701584727, -0.8828516073209223],
				  [0.1957627154676198, -0.0006519950679582, 0.1882218934706221, -0.1333245020495574, -0.6194690977235712, 0.1320420401250590, -0.0979484326519782, 0.0869066547652635, 0.1515520936725435, 0.0575871300734228, 0.8829946701584727, -0.8828516073209223],
				  [0.1957627154676198, -0.0006519950679582, 0.1882218934706221, -0.1333245020495574, -0.6194690977235712, 0.1320420401250590, -0.0979484326519782, 0.0869066547652635, 0.1515520936725435, 0.0575871300734228, 0.8829946701584727, -0.8828516073209223],
				  [0.1957627154676198, -0.0006519950679582, 0.1882218934706221, -0.1333245020495574, -0.6194690977235712, 0.1320420401250590, -0.0979484326519782, 0.0869066547652635, 0.1515520936725435, 0.0575871300734228, 0.8829946701584727, -0.8828516073209223],
				  [-1.2676314631743184, 1.0960968513720966, 0.8854462854465506, 1.2178978910852283, 1.4491341200593286, -1.2168245987604211, 1.1875392454748590, -1.1777456183509443, -1.2329289708702598, -1.1510121602423309, -0.4824610053326667, 0.4818976790520834],
				  [-1.2676314631743184, 1.0960968513720966, 0.8854462854465506, 1.2178978910852283, 1.4491341200593286, -1.2168245987604211, 1.1875392454748590, -1.1777456183509443, -1.2329289708702598, -1.1510121602423309, -0.4824610053326667, 0.4818976790520834],
				  [-1.2676314631743184, 1.0960968513720966, 0.8854462854465506, 1.2178978910852283, 1.4491341200593286, -1.2168245987604211, 1.1875392454748590, -1.1777456183509443, -1.2329289708702598, -1.1510121602423309, -0.4824610053326667, 0.4818976790520834]]
        des_fv = [[-1.3067242476681438, -1.6426237166935722, -1.7464021230713205, 1.3901236601260638, 0.8940296658581799, 1.6402653308200867, 0.9146044503381986, 0.9095899231990285, 1.7588591736737968, 1.4128924300753305, 0.9394862813693962, 1.5168366216927551],
				  [-1.3067242476681438, -1.6426237166935722, -1.7464021230713205, 1.3901236601260638, 0.8940296658581799, 1.6402653308200867, 0.9146044503381986, 0.9095899231990285, 1.7588591736737968, 1.4128924300753305, 0.9394862813693962, 1.5168366216927551],
				  [0.9223642422690362, 0.8218555443291123, 0.7577545897484805, -0.9066970703843535, -0.9481682366804809, 0.0028946925087050, 0.4911794272937892, 0.4936234323350660, -0.1315708313700105, -0.9015845840507067, -0.9486684856724428, 0.1143506173801624],
				  [0.9223642422690362, 0.8218555443291123, 0.7577545897484805, -0.9066970703843535, -0.9481682366804809, 0.0028946925087050, 0.4911794272937892, 0.4936234323350660, -0.1315708313700105, -0.9015845840507067, -0.9486684856724428, 0.1143506173801624],
				  [0.9223642422690362, 0.8218555443291123, 0.7577545897484805, -0.9066970703843535, -0.9481682366804809, 0.0028946925087050, 0.4911794272937892, 0.4936234323350660, -0.1315708313700105, -0.9015845840507067, -0.9486684856724428, 0.1143506173801624],
				  [0.9223642422690362, 0.8218555443291123, 0.7577545897484805, -0.9066970703843535, -0.9481682366804809, 0.0028946925087050, 0.4911794272937892, 0.4936234323350660, -0.1315708313700105, -0.9015845840507067, -0.9486684856724428, 0.1143506173801624],
				  [0.9223642422690362, 0.8218555443291123, 0.7577545897484805, -0.9066970703843535, -0.9481682366804809, 0.0028946925087050, 0.4911794272937892, 0.4936234323350660, -0.1315708313700105, -0.9015845840507067, -0.9486684856724428, 0.1143506173801624],
				  [-0.6661242386696298, -0.2746767627528053, -0.0986562341999207, 0.5844126772232056, 0.9842606172286825, -1.0983347080612154, -1.4283686790484511, -1.4290990026911261, -0.9532880634991853, 0.5607126867009591, 0.9547899552078075, -1.2018087767621075],
				  [-0.6661242386696298, -0.2746767627528053, -0.0986562341999207, 0.5844126772232056, 0.9842606172286825, -1.0983347080612154, -1.4283686790484511, -1.4290990026911261, -0.9532880634991853, 0.5607126867009591, 0.9547899552078075, -1.2018087767621075],
				  [-0.6661242386696298, -0.2746767627528053, -0.0986562341999207, 0.5844126772232056, 0.9842606172286825, -1.0983347080612154, -1.4283686790484511, -1.4290990026911261, -0.9532880634991853, 0.5607126867009591, 0.9547899552078075, -1.2018087767621075]]
        des_t = None
        des_gt = [1, 0, 0, 1] # Not normalized
        des_rp = None
        des_ro = None
        des_ts = None
        np.testing.assert_array_almost_equal(la, des_la, 13)
        np.testing.assert_array_almost_equal(av, des_av, 13)
        np.testing.assert_array_almost_equal(p, des_p, 12)
        np.testing.assert_array_almost_equal(v, des_v, 13)
        np.testing.assert_equal(t, des_t)
        np.testing.assert_array_almost_equal(fp, des_fp, 12)
        np.testing.assert_array_almost_equal(fv, des_fv, 13)
        np.testing.assert_array_almost_equal(gt, des_gt, 13)
        np.testing.assert_equal(r_p, des_rp)
        np.testing.assert_equal(r_o, des_ro)
        np.testing.assert_equal(ts, des_ts)

class TestQuadSDKDatasets(unittest.TestCase):
    """
    Test that the QuadSDK dataset class successfully
    processes and reads info from the dataset.
    """

    def setUp(self):
        # Get the paths to the URDF file and the dataset
        self.path_to_a1_urdf = Path(
            Path('.').parent, 'urdf_files', 'A1', 'a1.urdf').absolute()
        self.path_to_go2_quad_urdf = Path(
            Path('.').parent, 'urdf_files', 'Go2-Quad', 'go2.urdf').absolute()
        self.path_to_normal_sequence = Path(
            Path('.').parent, 'datasets', 'QuadSDK-A1Speed1.0').absolute()
        self.path_to_go2_seq = Path(Path('.').parent, 'datasets', 
                                    'QuadSDK-Go2-Flat-0.5-Mu50').absolute()

        # Set up the QuadSDK dataset
        self.dataset_hgnn_1 = QuadSDKDataset_A1Speed1_0_DEPRECATED(self.path_to_normal_sequence,
            self.path_to_a1_urdf, 'package://a1_description/',
                            'unitree_ros/robots/a1_description', 'heterogeneous_gnn', 1)
        
        # Set up the QuadSDK-Go2 Dataset
        self.dataset_go2_dynamics = QuadSDKDataset_Go2_Flat_Speed0_5_Mu_50(
            self.path_to_go2_seq, self.path_to_go2_quad_urdf, 'package://go2_description/', 
            '', 'dynamics', 1, normalize=False) 

    def test_process_and_load_data_at_ros_seq(self):
        """
        Make sure the data is processed and loaded properly from the file.
        """

        # ====================== Test A1 dataset ======================
        la, av, p, v, t, fp, fv, gt, r_p, r_o, ts = self.dataset_hgnn_1.load_data_at_dataset_seq(10000)
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
        des_gt = [64.74924447333427, 0, 0, 64.98097097053076]
        des_rp = [[-0.9077062285177977, 0.400808137988427, 0.25139755534263925]]
        des_ro = [[0.00807844275531408, 0.006594002480070749, -0.6310441334749343, -0.7756768396057802]]
        des_ts = [[27.374, 27.375, 27.374]]
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
        np.testing.assert_equal(ts, des_ts)

        # ====================== Test Go2 dataset ======================
        la, av, p, v, t, fp, fv, gt, r_p, r_o, ts = self.dataset_go2_dynamics.load_data_at_dataset_seq(19)
        des_la = [[1.04362010780786, 0.102924426497379, 9.69409887849744]]
        des_av = [[0.132802898128106, -0.136313764306598, 0.207648093420945]]
        des_p = [[-0.0186428812557580,	0.613793271947078,	1.32799863793224,	
                  0.0347470162543466,	0.698767277898859,	1.11862908126266,	
                  0.0198834628333486,	0.696377268016621,	1.13860830145294,	
                  0.00547654431396083,	0.574921459685768,	1.26371764626257]]
        des_v = [[-0.699186768494990,	-0.921542133257371,	0.277927285191015,
                  	0.612155995417224,	3.48897543944195,	3.66605185069497,	
                    0.380101618643802,	3.03484827328948,	3.21739436163926,	
                    -0.417627247095149,	-1.09821184134443,	0.416289250709227]]
        des_t = [[-5.04831081420462,	-0.817589258211806,	10.2766337684245,
                  	0.790579050967499,	-0.767835012490491,	-0.166955021638676,
                	-0.866199185499829,	-0.733417666986734,	-0.177643946203042,
                	4.58633681646792,	0.535467239477912,	10.7651478997751]]
        des_fp = None
        des_fv = None
        des_gt = [64.3055740713891,	0,	0,	70.2415205364494]
        des_rp = [[0.126104142299371,	0.0278440678211520,	0.262246743809873]]
        des_ro = [[-0.0112824130388665,	-0.0239457518490493,	0.00975197425090795,	0.999602024369006]]
        des_ts = [[30.1520000000000,	30.1510000000000,	30.1510000000000]]
    
        np.testing.assert_array_almost_equal(la, des_la, 14)
        np.testing.assert_array_almost_equal(av, des_av, 14)
        np.testing.assert_array_almost_equal(p, des_p, 13)
        np.testing.assert_array_almost_equal(v, des_v, 13)
        np.testing.assert_array_almost_equal(t, des_t, 13)
        np.testing.assert_equal(fp, des_fp)
        np.testing.assert_equal(fv, des_fv)
        np.testing.assert_array_almost_equal(gt, des_gt, 14)
        np.testing.assert_array_almost_equal(r_p, des_rp, 14)
        np.testing.assert_array_almost_equal(r_o, des_ro, 14)
        np.testing.assert_equal(ts, des_ts)


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

        la, av, p, v, t, fp, fv, gt, r_p, r_o, ts = self.dataset_mlp_1.load_data_at_dataset_seq(30000)
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
        des_gt = [1, 0, 0, 1]
        des_rp = None
        des_ro = None
        des_ts = None
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
        np.testing.assert_equal(ts, des_ts)

if __name__ == "__main__":
    unittest.main()
