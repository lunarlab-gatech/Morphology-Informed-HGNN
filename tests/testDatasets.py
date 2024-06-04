import unittest
from pathlib import Path
from grfgnn import CerberusStreetDataset, CerberusTrackDataset, Go1SimulatedDataset, FlexibleDataset, QuadSDKDataset
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
        self.go1_urdf = Path(
            Path('.').parent, 'urdf_files', 'Go1', 'go1.urdf').absolute()

        # Create the street dataset
        self.dataset_street_path = Path(
            Path(__file__).cwd(), 'datasets', 'cerberus_street')
        self.dataset_street = CerberusStreetDataset(self.dataset_street_path,
            self.a1_path,'package://a1_description/', 'unitree_ros/robots/a1_description', 'gnn')

        # Create the track dataset
        self.dataset_track_path = Path(
            Path(__file__).cwd(), 'datasets', 'cerberus_track')
        self.dataset_track = CerberusTrackDataset(self.dataset_track_path,
            self.a1_path, 'package://a1_description/', 'unitree_ros/robots/a1_description', 'gnn')

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
                self.go1_urdf, 'package://go1_description/', 'unitree_ros/robots/go1_description')
        self.assertEqual(
            "Invalid URDF: \"go1\". This dataset was collected on the \"a1\" robot. Please pass in the URDF file for the \"a1\" robot, not for the \"go1\" robot.",
            str(error.exception))
        with self.assertRaises(ValueError) as error:
            temp = CerberusTrackDataset(self.dataset_track_path, 
                self.go1_urdf, 'package://go1_description/', 'unitree_ros/robots/go1_description')
        self.assertEqual(
            "Invalid URDF: \"go1\". This dataset was collected on the \"a1\" robot. Please pass in the URDF file for the \"a1\" robot, not for the \"go1\" robot.",
            str(error.exception))

        # Make sure it detects when we pass an invalid model.
        with self.assertRaises(ValueError) as error:
            temp = CerberusStreetDataset(self.dataset_street_path,
                self.go1_urdf, 'package://go1_description/', 'unitree_ros/robots/go1_description', 'fake')
        self.assertEqual(
            "Parameter 'data_format' must be 'gnn', 'mlp', or 'heterogeneous_gnn'.",
            str(error.exception))

        # Make sure we get an error when we initialize a class that isn't meant
        # to be initialized.
        with self.assertRaises(NotImplementedError) as error:
            temp = FlexibleDataset(self.dataset_street_path, 
                    self.go1_urdf, 'package://go1_description/', 'unitree_ros/robots/go1_description')
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

class TestGo1SimulatedDataset(unittest.TestCase):
    """
    Test that the Go1 Simulated dataset successfully
    processes and reads the info for creating graph datasets.
    """

    def setUp(self):
        path_to_go1_urdf = Path(
            Path('.').parent, 'urdf_files', 'Go1', 'go1.urdf').absolute()
        path_to_xiong_simulated = Path(
            Path('.').parent, 'datasets', 'xiong_simulated').absolute()
        
        # Skip this test if the dataset isn't available
        # (Since we can't automatically download it from the Internet)
        if not path_to_xiong_simulated.exists():
            self.skipTest("QuadSDK-NormalSequence Dataset not available")

        # Set up the Go1 Simulated dataset
        model_type = 'heterogeneous_gnn'
        self.go1_sim_dataset = Go1SimulatedDataset(path_to_xiong_simulated,
            path_to_go1_urdf, 'package://go1_description/', 'unitree_ros/robots/go1_description', model_type)

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
        bj, jb, jj, fj, jf = self.go1_sim_dataset.robotGraph.get_edge_index_matrices()

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
        number_des = self.go1_sim_dataset.robotGraph.get_num_nodes()
        self.assertEqual(heteroData.num_nodes, number_des)

        # Check the node attributes
        base_x = np.array([[1]], dtype=np.float32)
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
        foot_x = np.array([[1], [1], [1], [1]], dtype=np.float32)
        np.testing.assert_array_equal(heteroData['base'].x.numpy(), base_x)
        np.testing.assert_array_equal(heteroData['joint'].x.numpy(), joint_x)
        np.testing.assert_array_equal(heteroData['foot'].x.numpy(), foot_x)

class TestQuadSDKDataset(unittest.TestCase):
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
        
        # Skip this test if the dataset isn't available
        # (Since we can't automatically download it from the Internet)
        if not path_to_normal_sequence.exists():
            self.skipTest("QuadSDK-A1Speed1.0 Dataset not available")

        # Set up the Go1 Simulated dataset
        model_type = 'heterogeneous_gnn'
        self.quad_sdk_dataset = QuadSDKDataset(path_to_normal_sequence,
            path_to_a1_urdf, 'package://a1_description/',
                            'unitree_ros/robots/a1_description', model_type)

    def test_load_data_at_ros_seq(self):
        """
        Make sure the data is loaded properly from the file, and that
        the angular acceleration and joint acceleration are calculated
        correctly.

        TODO: Verify that aa and ja aren't noisy.
        """

        # Make sure data is loaded properly
        la, av, p, v, t, gt, aa, ja = self.quad_sdk_dataset.load_data_at_dataset_seq(10000)
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
        des_gt = [64.74924447333427, 0, 0, 64.98097097053076]
        des_aa = [-4.5464778570261578, 0.3070732088085122, 2.46972771844984]
        des_ja = [3.4197760000000077, -0.23180600000000107, -0.7501079999999993, 4.2857779999999845,
                  17.511218000000017, 6.572418000000013, 12.040650000000008, -6.890848000000016,
                  11.833873999999955, 6.934245999999988, 2.426315999999995, 2.532692000000003]
        self.assertSequenceEqual(la, des_la)
        self.assertSequenceEqual(av, des_av)
        self.assertSequenceEqual(p, des_p)
        self.assertSequenceEqual(v, des_v)
        self.assertSequenceEqual(t, des_t)
        self.assertSequenceEqual(gt, des_gt)
        np.testing.assert_array_almost_equal(aa, des_aa, 12)
        np.testing.assert_array_almost_equal(ja, des_ja, 6)

    def test_load_data_sorted(self):


    def test_get_helper_heterogeneous_gnn(self):
        # Get the HeteroData graph
        heteroData: HeteroData = self.quad_sdk_dataset.get_helper_heterogeneous_gnn(14973)

        # Get the desired edge matrices
        bj, jb, jj, fj, jf = self.quad_sdk_dataset.robotGraph.get_edge_index_matrices()

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
        
        # Check the edge ATTRIBUTES
        self.assertTrue(False) # TODO: Write these tests

        # Check the labels
        labels_des = [33.086351867541104, 34.9074179562946, 33.245698338040306, 35.680794532426155]
        np.testing.assert_array_equal(heteroData.y.numpy(),
                                      np.array(labels_des, dtype=np.float32))
    
        # Check the foot node indices matching labels
        np.testing.assert_array_equal([0, 1, 2, 3], self.quad_sdk_dataset.get_foot_node_indices_matching_labels())

        # Check the number of nodes
        number_des = self.quad_sdk_dataset.robotGraph.get_num_nodes()
        self.assertEqual(heteroData.num_nodes, number_des)

        # Check the node attributes
        base_x = np.array([[0.051917279436213506, -0.3388206025562939, 10.431142687408848,
                            0.042171715199002736, 0.08603726244389077, -0.0004939939681741953]], dtype=np.float32)         
        joint_x = np.array([[0.10648644037804633, -0.0727984794236061, 2.76236804388072],
                            [0.41771433153947246, 0.2887144537724729, 1.2100160823330153],
                            [1.2560539486951967, 0.11197447543325509, 4.563897880420952],
                            [0.012585129280024532, -0.0630377445899227, -3.2944837117798227],
                            [0.7506705348026959, 0.29317922706218247, -1.2847036402444743],
                            [1.312375393869778, 0.19101350083753876, 6.045042601855347],
                            [0.0460617990056047, -0.07395783624617305, 3.283623688917122],
                            [0.7198842185898737, 0.4014568747126774, -1.2645530511487044],
                            [1.2232693367462089, 0.4080045806572458, 6.027049747813169],
                            [-0.15342791403898026, -0.018676861120108783, -2.040973443526994],
                            [0.3725592052364286, 0.47864302503484435, 1.3744733902657487],
                            [1.2010136473202389, 0.3749528264309599, 4.943340101162825]],
                           dtype=np.float32)
        foot_x = np.array([[1], [1], [1], [1]], dtype=np.float32)
        np.testing.assert_array_equal(heteroData['base'].x.numpy(), base_x)
        np.testing.assert_array_equal(heteroData['joint'].x.numpy(), joint_x)
        np.testing.assert_array_equal(heteroData['foot'].x.numpy(), foot_x)

if __name__ == '__main__':
    unittest.main()
