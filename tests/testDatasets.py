import unittest
from pathlib import Path
from grfgnn import RobotURDF, CerberusStreetDataset


class TestCerberusStreetDataset(unittest.TestCase):

    def setUp(self):
        # Load the A1 URDF file
        self.a1_path = Path(
            Path(__file__).cwd(), 'urdf_files', 'A1', 'a1.urdf').absolute()
        A1_URDF = RobotURDF(self.a1_path, 'package://a1_description/',
                            'unitree_ros/robots/a1_description', True)

        # Create the dataset
        self.dataset_path = Path(
            Path(__file__).cwd(), 'datasets', 'cerberus_street')
        dataset = CerberusStreetDataset(self.dataset_path, A1_URDF)

    def test_download_and_processing(self):
        """
        Check that the Dataset class successfully downloads and processes
        the data.
        """

        self.bag_path = self.dataset_path / 'raw' / 'street.bag'
        self.assertTrue(self.bag_path.is_file())
        self.processed_txt_path = self.dataset_path / 'processed' / 'processed.txt'
        self.assertTrue(self.processed_txt_path.is_file())


if __name__ == '__main__':
    unittest.main()
