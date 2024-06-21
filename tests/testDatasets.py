import unittest
from pathlib import Path
from grfgnn import CerberusStreetDataset, CerberusTrackDataset, Go1SimulatedDataset, FlexibleDataset, QuadSDKDataset_A1Speed1_0, QuadSDKDataset
from rosbags.highlevel import AnyReader
from torch_geometric.data import Data, HeteroData
import numpy as np
import torch

class TestFlexibleDatasets(unittest.TestCase):
    """
    Test that we can't initialize certain classes properly.
    """

    def test_initialization(self):
        # Get the paths to a URDF file and the datasets
        path_to_a1_urdf = Path(
            Path('.').parent, 'urdf_files', 'A1', 'a1.urdf').absolute()
        path_to_flexible = Path(Path('.').parent, 'datasets',
                                'Flexible').absolute()
        path_to_quad_sdk = Path(Path('.').parent, 'datasets',
                                'QuadSDK').absolute()

        # Try to create them, hope for an error
        with self.assertRaises(NotImplementedError):
            FlexibleDataset(path_to_flexible, path_to_a1_urdf, 'package://a1_description/',
                'unitree_ros/robots/a1_description', 'heterogeneous_gnn', 1)
        with self.assertRaises(NotImplementedError):
            QuadSDKDataset(path_to_quad_sdk, path_to_a1_urdf, 'package://a1_description/',
                'unitree_ros/robots/a1_description', 'heterogeneous_gnn', 1)

class TestQuadSDKDatasets(unittest.TestCase):
    """
    Test that the QuadSDK dataset successfully
    processes and reads the info for creating graph datasets.
    """

    def setUp(self):
        # Get the paths to the URDF file and the dataset
        path_to_a1_urdf = Path(
            Path('.').parent, 'urdf_files', 'A1', 'a1.urdf').absolute()
        path_to_normal_sequence = Path(
            Path('.').parent, 'datasets', 'QuadSDK-A1Speed1.0').absolute()

        # Set up the QuadSDK datasets
        self.dataset_hgnn_1 = QuadSDKDataset_A1Speed1_0(path_to_normal_sequence,
            path_to_a1_urdf, 'package://a1_description/',
                            'unitree_ros/robots/a1_description', 'heterogeneous_gnn', 1)
        self.dataset_gnn_1 = QuadSDKDataset_A1Speed1_0(path_to_normal_sequence,
            path_to_a1_urdf, 'package://a1_description/',
                            'unitree_ros/robots/a1_description', 'gnn', 1)
        self.dataset_mlp_1 = QuadSDKDataset_A1Speed1_0(path_to_normal_sequence,
            path_to_a1_urdf, 'package://a1_description/',
                            'unitree_ros/robots/a1_description', 'mlp', 1)
        self.dataset_hgnn_3 = QuadSDKDataset_A1Speed1_0(path_to_normal_sequence,
            path_to_a1_urdf, 'package://a1_description/',
                            'unitree_ros/robots/a1_description', 'heterogeneous_gnn', 3)
        self.dataset_gnn_3 = QuadSDKDataset_A1Speed1_0(path_to_normal_sequence,
            path_to_a1_urdf, 'package://a1_description/',
                            'unitree_ros/robots/a1_description', 'gnn', 3)
        self.dataset_mlp_3 = QuadSDKDataset_A1Speed1_0(path_to_normal_sequence,
            path_to_a1_urdf, 'package://a1_description/',
                            'unitree_ros/robots/a1_description', 'mlp', 3)

    def test_load_data_at_ros_seq(self):
        """
        Make sure the data is loaded properly from the file, and that
        the angular acceleration and joint acceleration are calculated
        correctly.
        """

        la, av, p, v, t, fp, fv, gt = self.dataset_hgnn_1.load_data_at_dataset_seq(10000)
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
        np.testing.assert_array_equal(la, des_la)
        np.testing.assert_array_equal(av, des_av)
        np.testing.assert_array_equal(p, des_p)
        np.testing.assert_array_equal(v, des_v)
        np.testing.assert_array_equal(t, des_t)
        np.testing.assert_array_equal(fp, des_fp)
        np.testing.assert_array_equal(fv, des_fv)
        np.testing.assert_array_equal(gt, des_gt)

    def test_load_data_sorted(self):
        """
        Test that the sorting to match the orders provided in self.foot_urdf_names
        and in self.joint_nodes_for_attributes.
        """
        la, av, p, v, t, fp, fv, gt = self.dataset_hgnn_1.load_data_sorted(10000)
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
        np.testing.assert_array_equal(la, des_la)
        np.testing.assert_array_equal(av, des_av)
        np.testing.assert_array_equal(p, des_p)
        np.testing.assert_array_equal(v, des_v)
        np.testing.assert_array_equal(t, des_t)
        np.testing.assert_array_equal(fp, des_fp)
        np.testing.assert_array_equal(fv, des_fv)
        np.testing.assert_array_equal(gt, des_gt)

    def test_get_helper_mlp(self):
        # Get the inputs and labels
        x, y = self.dataset_mlp_1.get_helper_mlp(9999)

        # Define the desired data
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

        # Test the values
        np.testing.assert_array_equal(y, des_y)
        np.testing.assert_array_almost_equal(x, des_x, 6)

    def test_get_helper_gnn(self):
        # Get the Data graph
        data: Data = self.dataset_gnn_1.get_helper_gnn(9999)

        # Define the desired data
        des_x = np.array([[1, 1, 1],
                        [0.11254642823264671, -1.2845189574751656, -0.6701594400127251],
                        [0.5695073200020913,  2.2337115054710917, -0.48841280756095506],
                        [0.9825683053175158, 3.4964483387811476, -0.1350813273560049],
                        [1, 1, 1],
                        [-0.16056788963386381, -1.015061543404335, -3.658930090797254],
                        [ 0.6448773402529877, 0.459564643757568,-3.858403899440098],
                        [1.1609664103261004, 0.15804277355899754, 12.475195605359755],
                        [1, 1, 1],
                        [0.22800622574234453, 1.020374615076573, 3.6351217639084576],
                        [0.4399285706508147, -0.34825271015763287, 4.456326408036115],
                        [1.1786520769378077, 0.3185087654826033,  7.829759255207876],
                        [1, 1, 1],
                        [-0.1931352599922853, 1.9489188274516005, 0.6111713969715354],
                        [ 0.4076711540455795, 1.3299772937985548, -0.11888726638996594],
                        [ 0.9424138768126973, 3.644698146547278, -0.24871924601280232],
                        [1, 1, 1]], dtype=np.float64)
        des_y = np.array([0, 64.74924447333427, 64.98097097053076, 0], dtype=np.float64)
        des_nodes = self.dataset_gnn_1.robotGraph.get_num_nodes()
        des_edge = self.dataset_gnn_1.robotGraph.get_edge_index_matrix()

        # Test the values
        np.testing.assert_array_equal(data.y, des_y)
        np.testing.assert_array_almost_equal(data.x, des_x, 6)
        self.assertEqual(data.num_nodes, des_nodes)
        np.testing.assert_array_equal(data.edge_index, des_edge)

    def test_get_helper_heterogeneous_gnn(self):
        # Get the HeteroData graph
        heteroData: HeteroData = self.dataset_hgnn_1.get_helper_heterogeneous_gnn(9999)

        # Test the desired edge matrices
        bj, jb, jj, fj, jf = self.dataset_hgnn_1.robotGraph.get_edge_index_matrices()
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
        bj_attr, jb_attr, jj_attr, fj_attr, jf_attr = self.dataset_hgnn_1.robotGraph.get_edge_attr_matrices()
        np.testing.assert_array_equal(heteroData['base', 'connect', 'joint'].edge_attr.numpy(), bj_attr)
        np.testing.assert_array_equal(heteroData['joint', 'connect', 'base'].edge_attr.numpy(), jb_attr)
        np.testing.assert_array_equal(heteroData['joint', 'connect', 'joint'].edge_attr.numpy(), jj_attr)
        np.testing.assert_array_equal(heteroData['foot', 'connect', 'joint'].edge_attr.numpy(), fj_attr)
        np.testing.assert_array_equal(heteroData['joint', 'connect', 'foot'].edge_attr.numpy(), jf_attr)

        # Check the labels
        labels_des = [0, 64.74924447333427, 64.98097097053076, 0]
        np.testing.assert_array_equal(heteroData.y.numpy(),
                                      np.array(labels_des, dtype=np.float64))

        # Check the foot node indices matching labels
        np.testing.assert_array_equal([0, 1, 2, 3], self.dataset_hgnn_1.get_foot_node_indices_matching_labels())

        # Check the number of nodes
        number_des = self.dataset_hgnn_1.robotGraph.get_num_nodes()
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
        np.testing.assert_array_almost_equal(heteroData['base'].x.numpy(), base_x, 12)
        np.testing.assert_array_almost_equal(heteroData['joint'].x.numpy(), joint_x, 6)
        np.testing.assert_array_equal(heteroData['foot'].x.numpy(), foot_x)

    def test_get(self):
        # Test that the get function reacts properly to dataset bounds.
        with self.assertRaises(IndexError):
            data = self.dataset_hgnn_1.get(-1)
        with self.assertRaises(IndexError):
            data = self.dataset_hgnn_1.get(17529)
        data = self.dataset_hgnn_1.get(0)
        data = self.dataset_hgnn_1.get(17528)

        # Test that it returns the proper values based on the model type
        x, y = self.dataset_mlp_1.get(0)
        self.assertEqual(type(x), torch.Tensor)
        self.assertEqual(type(y), torch.Tensor)

        data = self.dataset_gnn_1.get(0)
        self.assertEqual(type(data), Data)

        data = self.dataset_hgnn_1.get(0)
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
            data = self.dataset_mlp_3.get(-1)
        with self.assertRaises(IndexError):
            data = self.dataset_mlp_3.get(17527)
        data = self.dataset_mlp_3.get(0)
        data = self.dataset_mlp_3.get(17526)

        # ================================= MLP ==========================================
        # Get the output
        x_actual, y_actual = self.dataset_mlp_3.get(9997)

        # Calculated the desired x and y values
        xb2, yb2 = self.dataset_mlp_1.get_helper_mlp(9997)
        xb1, yb1 = self.dataset_mlp_1.get_helper_mlp(9998)
        x, y_des = self.dataset_mlp_1.get_helper_mlp(9999)
        x_comb = torch.stack((xb2, xb1, x), 0)
        x_des = torch.flatten(torch.transpose(x_comb, 0, 1), 0, 1)

        # Test the values
        np.testing.assert_equal(x_actual.numpy(), x_des.numpy())
        np.testing.assert_equal(y_actual.numpy(), y_des.numpy())

        # ================================= GNN ==========================================

        # Get the output
        data_actual: Data = self.dataset_gnn_3.get_helper_gnn(9997)

        # Check the labels
        labels_des = [0, 64.74924447333427, 64.98097097053076, 0]
        np.testing.assert_array_equal(data_actual.y.numpy(), np.array(labels_des, dtype=np.float64))

        # Get desired node attributes
        datab2 = self.dataset_gnn_1.get_helper_gnn(9997)
        datab1 = self.dataset_gnn_1.get_helper_gnn(9998)
        data = self.dataset_gnn_1.get_helper_gnn(9999)
        x_des = torch.cat((datab2.x[:,0].unsqueeze(1), datab1.x[:,0].unsqueeze(1), data.x[:,0].unsqueeze(1),
                            datab2.x[:,1].unsqueeze(1), datab1.x[:,1].unsqueeze(1), data.x[:,1].unsqueeze(1),
                            datab2.x[:,2].unsqueeze(1), datab1.x[:,2].unsqueeze(1), data.x[:,2].unsqueeze(1)), 1)
        y_des = data.y

        # Check the values
        np.testing.assert_equal(data_actual.x.numpy(), x_des.numpy())
        np.testing.assert_equal(data_actual.y.numpy(), y_des.numpy())

        # ================================= Heterogeneous GNN ==========================================

        # Get the HeteroData graph
        heteroData: HeteroData = self.dataset_hgnn_3.get_helper_heterogeneous_gnn(9997)

        # Get the desired node attributes
        hDatab2 = self.dataset_hgnn_1.get_helper_heterogeneous_gnn(9997)
        hDatab1 = self.dataset_hgnn_1.get_helper_heterogeneous_gnn(9998)
        hData = self.dataset_hgnn_1.get_helper_heterogeneous_gnn(9999)
        base_x_cat = torch.cat((hDatab2['base'].x, hDatab1['base'].x, hData['base'].x), 0)
        base_x_des = torch.flatten(torch.transpose(base_x_cat, 0, 1), 0).unsqueeze(0)
        joint_x_des = torch.cat((hDatab2['joint'].x[:,0].unsqueeze(1), hDatab1['joint'].x[:,0].unsqueeze(1), hData['joint'].x[:,0].unsqueeze(1),
                                 hDatab2['joint'].x[:,1].unsqueeze(1), hDatab1['joint'].x[:,1].unsqueeze(1), hData['joint'].x[:,1].unsqueeze(1),
                                 hDatab2['joint'].x[:,2].unsqueeze(1), hDatab1['joint'].x[:,2].unsqueeze(1), hData['joint'].x[:,2].unsqueeze(1)), 1)
        foot_x = hData['foot'].x
        y = hData.y

        # Check the values
        np.testing.assert_array_almost_equal(heteroData['base'].x.numpy(), base_x_des.numpy(), 12)
        np.testing.assert_array_almost_equal(heteroData['joint'].x.numpy(), joint_x_des.numpy(), 6)
        np.testing.assert_array_equal(heteroData['foot'].x.numpy(), foot_x.numpy())
        np.testing.assert_array_equal(heteroData.y.numpy(), y.numpy())


if __name__ == "__main__":
    unittest.main()
