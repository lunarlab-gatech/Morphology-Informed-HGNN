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
        des_la = [-0.06452160178213015, -0.366493877667443, 9.715652148737323]
        des_av = [-0.0017398309484803294, -0.011335050676391335, 1.2815129213608234]
        des_p = [-0.16056788963386381,  0.6448773402529877, 1.1609664103261004,
                 0.11254642823264671, 0.5695073200020913, 0.9825683053175158,
                 -0.1931352599922853,  0.4076711540455795,  0.9424138768126973,
                  0.22800622574234453, 0.4399285706508147, 1.1786520769378077]
        des_v = [-1.015061543404335, 0.459564643757568, 0.15804277355899754,
                 -1.2845189574751656, 2.2337115054710917, 3.4964483387811476,
                  1.9489188274516005, 1.3299772937985548, 3.644698146547278,
                  1.020374615076573, -0.34825271015763287, 0.3185087654826033]
        des_t = [-3.658930090797254, -3.858403899440098, 12.475195605359755,
                 -0.6701594400127251,-0.48841280756095506, -0.1350813273560049,
                 0.6111713969715354, -0.11888726638996594, -0.24871924601280232,
                 3.6351217639084576, 4.456326408036115, 7.829759255207876]
        des_fp = None
        des_fv = None
        des_gt = [0, 64.74924447333427, 64.98097097053076, 0]
        des_rp = [-0.9077062285177977, 0.400808137988427, 0.25139755534263925]
        des_ro = [0.00807844275531408, 0.006594002480070749, -0.6310441334749343, -0.7756768396057802]
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
        des_la = [0.146148949861527,	0.0223360173404217,	9.58438873291016] 
        des_av = [0.00771053740754724,	0.0315540358424187,	0.161459520459175]
        des_p = [0.0581750869750977,	-0.498899102210999,	1.01744437217712,	
                 0.000190734863281250,	-0.625548958778381,	1.45497155189514,	
                 0.00858306884765625, -0.650104880332947,	1.49907255172730,	
                 0.0394830703735352,	-0.420838713645935,	0.846460103988648]
        des_v = [0.0793685913085938,	-1.38095092773438,	0.0102071752771735,	
                 0.142860412597656,	8.93650817871094,	-9.21489906311035,	
                 -1.60317611694336,	9.79365158081055,	-9.19449234008789,	
                 0.301589965820313,	-1.47618865966797,	-0.418396085500717]
        des_t = None
        des_fp = [-0.00335261225700378,	-0.0453704893589020,	-0.356130868196487,	
                  0.0214422345161438,	0.0660574287176132,	-0.301095396280289,	
                  0.0198653340339661,	-0.0634637400507927,	-0.295771747827530,	
                  -0.00486903637647629,	0.0804890319705010,	-0.365474313497543]
        des_fv = [-0.485595673322678,	0.0279390625655651,	0.00200577825307846,	
                  1.47740077972412,	0.0432308465242386,	-1.12423670291901,	
                  1.70655310153961,	-0.464293301105499,	-1.04942762851715,	
                  -0.618086814880371,	0.111269362270832, -0.00220335274934769]
        des_gt = [1, 0,	0, 1]
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
        des_la = [-0.06452160178213015, -0.366493877667443, 9.715652148737323]
        des_av = [-0.0017398309484803294, -0.011335050676391335, 1.2815129213608234]
        des_p = [-0.16056788963386381,  0.6448773402529877, 1.1609664103261004, -0.1931352599922853,
                 0.4076711540455795,  0.9424138768126973,  0.11254642823264671, 0.5695073200020913,
                 0.9825683053175158, 0.22800622574234453, 0.4399285706508147, 1.1786520769378077]
        des_v = [-1.015061543404335, 0.459564643757568, 0.15804277355899754, 1.9489188274516005,
                  1.3299772937985548, 3.644698146547278, -1.2845189574751656, 2.2337115054710917,
                  3.4964483387811476, 1.020374615076573, -0.34825271015763287, 0.3185087654826033]
        des_t = [-3.658930090797254, -3.858403899440098, 12.475195605359755, 0.6111713969715354,
                 -0.11888726638996594, -0.24871924601280232, -0.6701594400127251,-0.48841280756095506,
                 -0.1350813273560049, 3.6351217639084576, 4.456326408036115, 7.829759255207876]
        des_fp = None
        des_fv = None
        des_gt = [64.74924447333427, 0, 0, 64.98097097053076]
        des_rp = [-0.9077062285177977, 0.400808137988427, 0.25139755534263925]
        des_ro = [0.00807844275531408, 0.006594002480070749, -0.6310441334749343, -0.7756768396057802]
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
        des_la = [0.146148949861527,	0.0223360173404217,	9.58438873291016] 
        des_av = [0.00771053740754724,	0.0315540358424187,	0.161459520459175]
        des_p = [0.0581750869750977,	-0.498899102210999,	1.01744437217712,	
                 0.000190734863281250,	-0.625548958778381,	1.45497155189514,	
                 0.00858306884765625, -0.650104880332947,	1.49907255172730,	
                 0.0394830703735352,	-0.420838713645935,	0.846460103988648]
        des_v = [0.0793685913085938,	-1.38095092773438,	0.0102071752771735,	
                 0.142860412597656,	8.93650817871094,	-9.21489906311035,	
                 -1.60317611694336,	9.79365158081055,	-9.19449234008789,	
                 0.301589965820313,	-1.47618865966797,	-0.418396085500717]
        des_t = None
        des_fp = [-0.00335261225700378,	-0.0453704893589020,	-0.356130868196487,	
                  0.0214422345161438,	0.0660574287176132,	-0.301095396280289,	
                  0.0198653340339661,	-0.0634637400507927,	-0.295771747827530,	
                  -0.00486903637647629,	0.0804890319705010,	-0.365474313497543]
        des_fv = [-0.485595673322678,	0.0279390625655651,	0.00200577825307846,	
                  1.47740077972412,	0.0432308465242386,	-1.12423670291901,	
                  1.70655310153961,	-0.464293301105499,	-1.04942762851715,	
                  -0.618086814880371,	0.111269362270832, -0.00220335274934769]
        des_gt = [1, 0,	0, 1]
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
