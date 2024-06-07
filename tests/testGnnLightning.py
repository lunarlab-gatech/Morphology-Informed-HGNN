import unittest
from pathlib import Path
from grfgnn import QuadSDKDataset_A1Speed1_0, QuadSDKDataset_A1Speed0_5, QuadSDKDataset_A1Speed1_5FlippedOver
from grfgnn.gnnLightning import train_model, evaluate_model
import torch
from torch.utils.data import random_split
import numpy as np

class TestGnnLighting(unittest.TestCase):
    """
    Test the classes and functions found in the
    gnnLightning.py file.
    """

    def setUp(self):
        """
        Setup the necessary datasets for use in testing later.
        """
        # Setup a random generator
        self.rand_gen = torch.Generator().manual_seed(10341885)

        # Initalize the datasets
        path_to_urdf = Path('urdf_files', 'A1', 'a1.urdf').absolute()
        path_to_quad_sdk = Path(
            Path('.').parent, 'datasets', 'QuadSDK-A1Speed1.0').absolute()

        self.dataset_mlp = QuadSDKDataset_A1Speed1_0(
            path_to_quad_sdk, path_to_urdf, 'package://a1_description/',
            'unitree_ros/robots/a1_description', 'mlp')
        self.dataset_gnn = QuadSDKDataset_A1Speed1_0(
            path_to_quad_sdk, path_to_urdf, 'package://a1_description/',
            'unitree_ros/robots/a1_description', 'gnn')
        self.dataset_hgnn = QuadSDKDataset_A1Speed1_0(
            path_to_quad_sdk, path_to_urdf, 'package://a1_description/',
            'unitree_ros/robots/a1_description', 'heterogeneous_gnn')
        self.dataset_hgnn_3 = QuadSDKDataset_A1Speed1_0(
            path_to_quad_sdk, path_to_urdf, 'package://a1_description/',
            'unitree_ros/robots/a1_description', 'heterogeneous_gnn', 3)

        # Put them into an array for easy testing over all of them
        self.models = [self.dataset_mlp, self.dataset_gnn, self.dataset_hgnn, self.dataset_hgnn_3]

    def test_train_and_eval_model(self):
        """
        Make sure that the train model function runs and
        finishes successfully, and test that we can evaluate
        each model as well without a crash.
        """

        # For each model
        for model in self.models:
            # Train on the model
            train_dataset, val_dataset, test_dataset = random_split(
                model, [0.7, 0.2, 0.1], generator=self.rand_gen)
            path_to_ckpt_folder = train_model(train_dataset, val_dataset, test_dataset,
                                              testing_mode=True, disable_logger=True)

            # Make sure five models were saved
            models = sorted(Path('.', path_to_ckpt_folder).glob(("epoch=*")))
            self.assertEqual(len(models), 5)

            # Predict with the model
            pred, labels = evaluate_model(models[0], test_dataset)

            # Assert the sizes of the results match
            self.assertEqual(pred.shape[0], len(test_dataset.indices))
            self.assertEqual(pred.shape[1], 4)
            self.assertEqual(labels.shape[0], len(test_dataset.indices))
            self.assertEqual(labels.shape[1], 4)

if __name__ == "__main__":
    unittest.main()
