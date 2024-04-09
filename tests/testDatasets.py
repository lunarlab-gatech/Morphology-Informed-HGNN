import unittest
from pathlib import Path
from grfgnn import RobotURDF, CerberusStreetDataset, CerberusTrackDataset
from rosbags.highlevel import AnyReader


class TestCerberusDatasets(unittest.TestCase):
    """
    Test that the Cerberus Dataset classes properly download,
    process, and read the info for creating graph datasets. We
    used webviz.io in order to inspect the rosbag and genereate 
    regression test cases.
    """

    def setUp(self):
        # Load the A1 URDF file
        self.a1_path = Path(
            Path(__file__).cwd(), 'urdf_files', 'A1', 'a1.urdf').absolute()
        A1_URDF = RobotURDF(self.a1_path, 'package://a1_description/',
                            'unitree_ros/robots/a1_description', True)

        # Create the street dataset
        self.dataset_street_path = Path(
            Path(__file__).cwd(), 'datasets', 'cerberus_street')
        self.dataset_street = CerberusStreetDataset(self.dataset_street_path,
                                                    A1_URDF)

        # Create the track dataset
        self.dataset_track_path = Path(
            Path(__file__).cwd(), 'datasets', 'cerberus_track')
        self.dataset_track = CerberusTrackDataset(self.dataset_track_path,
                                                  A1_URDF)

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
        p, v, e = self.dataset_street.load_data_at_ros_seq(1597)
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
        self.assertSequenceEqual(p, des_p)
        self.assertSequenceEqual(v, des_v)
        self.assertSequenceEqual(e, des_e)

        # Test the last entry in the street dataset
        p, v, e = self.dataset_street.load_data_at_ros_seq(264904)
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
        self.assertSequenceEqual(p, des_p)
        self.assertSequenceEqual(v, des_v)
        self.assertSequenceEqual(e, des_e)

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

    # TODO: Add test cases for get() method
    # TODO: Add test cases for new dataset class methods


if __name__ == '__main__':
    unittest.main()
