import unittest
from pathlib import Path
from grfgnn import RobotURDF, CerberusStreetDataset


class TestCerberusStreetDataset(unittest.TestCase):
    """
    Test that the Cerberus Street Dataset class properly downloads,
    processes, and reads the info for creating graph datasets. We
    used webviz.io in order to inspect the rosbag and genereate 
    regression test cases.
    """

    def setUp(self):
        # Load the A1 URDF file
        self.a1_path = Path(
            Path(__file__).cwd(), 'urdf_files', 'A1', 'a1.urdf').absolute()
        A1_URDF = RobotURDF(self.a1_path, 'package://a1_description/',
                            'unitree_ros/robots/a1_description', True)

        # Create the dataset
        self.dataset_path = Path(
            Path(__file__).cwd(), 'datasets', 'cerberus_street')
        self.dataset = CerberusStreetDataset(self.dataset_path, A1_URDF)
        self.dataset_only_first_100 = CerberusStreetDataset(
            self.dataset_path, A1_URDF, 100)

    def test_download_and_processing(self):
        """
        Check that the Dataset class successfully downloads and processes
        the data, and that it correctly finds the first sequence number.
        """

        self.bag_path = self.dataset_path / 'raw' / 'data.bag'
        self.processed_txt_start_path = self.dataset_path / 'processed' / '1597.txt'
        self.processed_txt_end_path = self.dataset_path / 'processed' / '264904.txt'
        self.processed_info_path = self.dataset_path / 'processed' / 'info.txt'

        self.assertTrue(self.bag_path.is_file())
        self.assertTrue(self.processed_txt_start_path.is_file())
        self.assertTrue(self.processed_txt_end_path.is_file())
        self.assertTrue(self.processed_info_path.is_file())
        self.assertEqual(1597, self.dataset.first_index)

    def test_get_index_of_ros_name(self):
        """
        Test that the ROS joint name to index mapping is correct.
        """

        self.assertEqual(4, self.dataset.get_index_of_ros_name('FR1'))
        self.assertEqual(9, self.dataset.get_index_of_ros_name('RR0'))
        self.assertEqual(14, self.dataset.get_index_of_ros_name('RL_foot'))
        with self.assertRaises(IndexError):
            self.dataset.get_index_of_ros_name("Non-existant")

    def test_length(self):
        """
        Test that the length is properly loaded.
        """
        self.assertEqual(263308, self.dataset.len())

        # Test the length for the dataset where we only use
        # the first 100
        self.assertEqual(100, self.dataset_only_first_100.len())

    def test_load_data_at_ros_seq(self):
        """
        Test that we can accurately find the information stored 
        at a specific ros sequence number.
        """

        # Test the first entry in the dataset
        p, v, e = self.dataset.load_data_at_ros_seq(1597)
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

        # Test the last entry in the dataset
        p, v, e = self.dataset.load_data_at_ros_seq(264904)
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

        # Test with some out of bounds sequence numbers
        with self.assertRaises(OSError):
            self.dataset.load_data_at_ros_seq(1596)
        with self.assertRaises(OSError):
            self.dataset.load_data_at_ros_seq(264905)

    # TODO: Add test cases for track dataset
    # TODO: Add test cases to make sure that the names of the indices of
    # both dataset match up.


if __name__ == '__main__':
    unittest.main()
