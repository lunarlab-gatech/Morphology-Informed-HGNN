import unittest
from pathlib import Path
from grfgnn import NormalRobotGraph, HeterogeneousRobotGraph, CerberusStreetDataset, CerberusTrackDataset, CerberusCampusDataset, Go1SimulatedDataset, FlexibleDataset
from rosbags.highlevel import AnyReader
import os
from torch_geometric.data import HeteroData
import numpy as np


class TestCerberusDatasets(unittest.TestCase):
    """
    Test that the Cerberus Dataset classes download,
    process, and read the info for creating graph datasets. We
    used webviz.io in order to inspect the rosbag and genereate 
    regression test cases.
    """

    def setUp(self):
        # Load the URDF files
        self.a1_path = Path(
            Path(__file__).cwd(), 'urdf_files', 'A1', 'a1.urdf').absolute()
        self.go1_path = Path(
            Path(__file__).cwd(), 'urdf_files', 'Go1', 'go1.urdf').absolute()
        self.A1_URDF = NormalRobotGraph(self.a1_path,
                                        'package://a1_description/',
                                        'unitree_ros/robots/a1_description',
                                        True)
        self.GO1_URDF = NormalRobotGraph(self.go1_path,
                                         'package://go1_description/',
                                         'unitree_ros/robots/go1_description',
                                         True)

        # Create the street dataset
        self.dataset_street_path = Path(
            Path(__file__).cwd(), 'datasets', 'cerberus_street')
        self.dataset_street = CerberusStreetDataset(self.dataset_street_path,
                                                    self.A1_URDF, 'gnn')

        # Create the track dataset
        self.dataset_track_path = Path(
            Path(__file__).cwd(), 'datasets', 'cerberus_track')
        self.dataset_track = CerberusTrackDataset(self.dataset_track_path,
                                                  self.A1_URDF, 'gnn')

        # Create lists to hold the datasets and paths we want to loop through
        self.dataset_path_list = [
            self.dataset_street_path, self.dataset_track_path
        ]
        self.dataset_list = [self.dataset_street, self.dataset_track]

        # Ground truth sequence numbers and lengths
        self.first_seq_nums = [1597, 2603]
        self.last_seq_nums = [264904, 283736]
        self.lengths = [263308, 281134]

    def test_download_and_processing(self):
        """
        Check that the Dataset class successfully downloads and processes
        the data, and that it correctly finds the first sequence number.
        """

        for i, dataset in enumerate(self.dataset_list):
            path = self.dataset_path_list[i]
            self.bag_path = path / 'raw' / 'data.bag'
            self.processed_info_path = path / 'processed' / 'info.txt'

            self.assertTrue(self.bag_path.is_file())
            self.assertTrue(self.processed_info_path.is_file())
            self.assertEqual(self.first_seq_nums[i], dataset.first_index)

    def test_get_index_of_ros_name(self):
        """
        Test that the ROS joint name to index mapping is correct.
        """

        self.assertEqual(4, self.dataset_street.get_index_of_ros_name('FR1'))
        self.assertEqual(9, self.dataset_street.get_index_of_ros_name('RR0'))
        self.assertEqual(14,
                         self.dataset_street.get_index_of_ros_name('RL_foot'))
        with self.assertRaises(IndexError):
            self.dataset_street.get_index_of_ros_name("Non-existant")

    def test_length(self):
        """
        Test that the length is properly loaded.
        """
        for i, dataset in enumerate(self.dataset_list):
            self.assertEqual(self.lengths[i], dataset.len())

    def test_load_data_at_ros_seq(self):
        """
        Test that we can accurately find the information stored 
        at a specific ros sequence number.
        """

        # Test the first entry in the street dataset
        p, v, e, l = self.dataset_street.load_data_at_ros_seq(1597)
        des_p = [
            0.19145259261131287, 1.1113770008087158, -2.5523500442504883,
            -0.32268381118774414, 1.0603430271148682, -2.4854280948638916,
            0.18896619975566864, 1.1511175632476807, -2.5309417247772217,
            -0.1835719794034958, 1.142857551574707, -2.535998821258545, 0, 0,
            0, 0
        ]
        des_v = [
            -0.018887361511588097, -0.07383241504430771, 0.0017170329811051488,
            -0.10731455683708191, 0.020604396238923073, 0.0223214291036129,
            0.007726647891104221, 0.13993819057941437, 0.0008585164905525744,
            0.03863323852419853, -0.006009615492075682, 0.0034340659622102976,
            1, 1, 1, 1
        ]
        des_e = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 48.2, 43.8, 25.8, 24.6]
        des_l = [48.2, 43.8, 25.8, 24.6]
        self.assertSequenceEqual(p, des_p)
        self.assertSequenceEqual(v, des_v)
        self.assertSequenceEqual(e, des_e)
        self.assertSequenceEqual(l, des_l)

        # Test the last entry in the street dataset
        p, v, e, l = self.dataset_street.load_data_at_ros_seq(264904)
        des_p = [
            0.13228477537631989, 0.7995240688323975, -1.3597643375396729,
            -0.10687294602394104, 0.8143579959869385, -1.5252572298049927,
            0.05937854200601578, 0.674445629119873, -1.3225526809692383,
            -0.06970340758562088, 0.6872990131378174, -1.4055731296539307, 0,
            0, 0, 0
        ]
        des_v = [
            0.8250343203544617, 0.32451924681663513, -0.2652815878391266,
            -0.7915521264076233, 0.22149725258350372, -0.6756525039672852,
            0.276442289352417, -0.029189560562372208, -0.41809752583503723,
            -0.2970466911792755, 0.07211538404226303, -0.7228708863258362, 1,
            1, 1, 1
        ]
        des_e = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 98.6, 164.8, 69.2, 69.6]
        des_l = [98.6, 164.8, 69.2, 69.6]
        self.assertSequenceEqual(p, des_p)
        self.assertSequenceEqual(v, des_v)
        self.assertSequenceEqual(e, des_e)
        self.assertSequenceEqual(l, des_l)

        # TODO: Add specific test case for track dataset

        # Test that the out of bounds sequence numbers throw an exception,
        # while those within bounds don't.
        for i, dataset in enumerate(self.dataset_list):
            with self.assertRaises(OSError):
                dataset.load_data_at_ros_seq(self.first_seq_nums[i] - 1)
            dataset.load_data_at_ros_seq(self.first_seq_nums[i])
            dataset.load_data_at_ros_seq(self.last_seq_nums[i])
            with self.assertRaises(OSError):
                dataset.load_data_at_ros_seq(self.last_seq_nums[i] + 1)

    def test_all_name_indices_line_up(self):
        """
        This method makes sure that the name array provided in the messages of all
        supported datasets match each other. This assumption is critical, as
        the get_ground_truth_label_indices() method only works if this assumption
        holds.
        
        If this test fails, then the developers will need to implement custom 
        self.ros_name_in_index values per Dataset child class.
        """

        names_lists = []
        for i, dataset in enumerate(self.dataset_list):
            # Set up a reader to read the rosbag
            topic = '/hardware_a1/joint_foot'
            path_to_bag = Path(self.dataset_path_list[i], 'raw', 'data.bag')
            self.reader = AnyReader([Path(path_to_bag)])
            self.reader.open()
            self.joint_gen = self.reader.messages(connections=[
                x for x in self.reader.connections if x.topic == topic
            ])

            # Extract the list of ros names from the first ROS message
            for connection, _timestamp, rawdata in self.joint_gen:
                msg = self.reader.deserialize(rawdata, connection.msgtype)
                names_lists.append(msg.name)
                break

        # Need at least 2 datasets for this test to make sense
        self.assertTrue(len(names_lists) > 1)

        # Assert that the lists are all identical
        for i in range(0, len(names_lists) - 1):
            self.assertSequenceEqual(names_lists[i], names_lists[i + 1])

    def test_error_messages(self):
        """
        Make sure all of the proper error messages are thrown when we expect them.
        """

        # Makes sure the Dataset class properly detects when it's been passed
        # the wrong URDF file.
        with self.assertRaises(ValueError) as error:
            temp = CerberusStreetDataset(self.dataset_street_path,
                                         self.GO1_URDF)
        self.assertEqual(
            "Invalid URDF: \"go1\". This dataset was collected on the \"a1\" robot. Please pass in the URDF file for the \"a1\" robot, not for the \"go1\" robot.",
            str(error.exception))
        with self.assertRaises(ValueError) as error:
            temp = CerberusTrackDataset(self.dataset_track_path, self.GO1_URDF)
        self.assertEqual(
            "Invalid URDF: \"go1\". This dataset was collected on the \"a1\" robot. Please pass in the URDF file for the \"a1\" robot, not for the \"go1\" robot.",
            str(error.exception))

        # Make sure it detects when we pass an invalid model.
        with self.assertRaises(ValueError) as error:
            temp = CerberusStreetDataset(self.dataset_street_path,
                                         self.GO1_URDF, 'fake')
        self.assertEqual(
            "Parameter 'data_format' must be 'gnn', 'mlp', or 'heterogeneous_gnn'.",
            str(error.exception))

        # Make sure we get an error when we initialize a class that isn't meant
        # to be initialized.
        with self.assertRaises(NotImplementedError) as error:
            temp = FlexibleDataset(self.dataset_street_path, self.GO1_URDF)
        self.assertEqual(
            "Don't call this class directly, but use one of \
        the child classes in order to choose which dataset \
        sequence you want to load.", str(error.exception))

    def test_get_expected_urdf_name(self):
        """
        Make sure that the URDF method is properly set.
        """

        self.assertEqual("a1", self.dataset_street.get_expected_urdf_name())
        self.assertEqual("a1", self.dataset_track.get_expected_urdf_name())

    # TODO: Add test cases for get() method
    # TODO: Add test cases for new dataset class methods


@unittest.skipIf(
    os.getenv('RUN_LOCAL_TESTS', 'False') != 'True',
    "Not running tests on datasets that can't be downloaded")
class TestGo1SimulatedDataset(unittest.TestCase):
    """
    Test that the Go1 Simulated dataset successfully
    processes and reads the info for creating graph datasets.
    """

    def setUp(self):
        # Set up the Go1 Simulated dataset
        path_to_go1_urdf = Path(
            Path('.').parent, 'urdf_files', 'Go1', 'go1.urdf').absolute()
        path_to_xiong_simulated = Path(
            Path('.').parent, 'datasets', 'xiong_simulated').absolute()
        self.GO1_graph = HeterogeneousRobotGraph(
            path_to_go1_urdf, 'package://go1_description/',
            'unitree_ros/robots/go1_description', True)
        model_type = 'gnn'
        self.go1_sim_dataset = Go1SimulatedDataset(path_to_xiong_simulated,
                                                   self.GO1_graph, model_type)

    def test_load_data_at_ros_seq(self):
        """
        Make sure the data from the 100 rosbags are correctly given sequence
        numbers for the Go1 Simulated dataset.
        """

        # Make sure data is loaded properly
        p, v, t, gt = self.go1_sim_dataset.load_data_at_ros_seq(181916)
        des_p = [
            0.09840758, 1.2600079, -2.0489328, -0.10457229, 0.7454401,
            -1.8136898, 0.04816602, 0.91943306, -1.9169945, -0.0340704,
            1.2415031, -2.0689785
        ]
        des_v = [
            0.6230268, -5.9481835, -5.3682613, -0.26720884, 6.455389,
            -1.3378538, -0.00086710247, 6.2338834, -0.5447279, 0.6295713,
            -4.582517, -7.406303
        ]
        des_t = [
            -0.4899872, -3.0470033, 0.363765, 5.630082, 5.196147, 11.241279,
            -1.8939179, 4.083751, 16.447073, -1.3105631, -0.7087057, 0.13933142
        ]
        des_gt = [-0.0063045006, 55.528183, 83.40855, -0.006935571]
        self.assertSequenceEqual(p, des_p)
        self.assertSequenceEqual(v, des_v)
        self.assertSequenceEqual(t, des_t)
        self.assertSequenceEqual(gt, des_gt)

    def test_get_helper_heterogeneous_gnn(self):
        # Get the HeteroData graph
        heteroData: HeteroData = self.go1_sim_dataset.get_helper_heterogeneous_gnn(
            181916)
        # Get the desired edge matrices
        bj, jb, jj, fj, jf = self.GO1_graph.get_edge_index_matrices()

        # Make sure they match
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

        # Check the labels
        labels_des = [-0.0063045006, 55.528183, 83.40855, -0.006935571]
        np.testing.assert_array_equal(heteroData.y.numpy(),
                                      np.array(labels_des, dtype=np.float32))

        # Check the number of nodes
        number_des = self.GO1_graph.get_num_nodes()
        self.assertEqual(heteroData.num_nodes, number_des)

        # Check the node attributes
        base_x = np.array([[]], dtype=np.float32)
        joint_x = np.array([[0.09840758, 0.6230268, -0.4899872],
                            [1.2600079, -5.9481835, -3.0470033],
                            [-2.0489328, -5.3682613, 0.363765],
                            [-0.10457229, -0.26720884, 5.630082],
                            [0.7454401, 6.455389, 5.196147],
                            [-1.8136898, -1.3378538, 11.241279],
                            [0.04816602, -0.00086710247, -1.8939179],
                            [0.91943306, 6.2338834, 4.083751],
                            [-1.9169945, -0.5447279, 16.447073],
                            [-0.0340704, 0.6295713, -1.3105631],
                            [1.2415031, -4.582517, -0.7087057],
                            [-2.0689785, -7.406303, 0.13933142]],
                           dtype=np.float32)
        foot_x = np.array([[], [], [], []], dtype=np.float32)
        np.testing.assert_array_equal(heteroData['base'].x.numpy(), base_x)
        np.testing.assert_array_equal(heteroData['joint'].x.numpy(), joint_x)
        np.testing.assert_array_equal(heteroData['foot'].x.numpy(), foot_x)


if __name__ == '__main__':
    unittest.main()
