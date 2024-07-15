import unittest
from pathlib import Path
from grfgnn import QuadSDKDataset_A1Speed1_0
from grfgnn.gnnLightning import gnn_classification_output
from grfgnn.datasets_py.LinTzuYaunDataset import LinTzuYaunDataset_asphalt_road
from grfgnn.gnnLightning import train_model, evaluate_model, get_foot_node_outputs_gnn, get_foot_node_labels_gnn, Heterogeneous_GNN_Lightning, Base_Lightning
import torch
from torch.utils.data import random_split
import numpy as np
from torch_geometric.loader import DataLoader

class TestGnnLightning(unittest.TestCase):
    """
    Test the classes and functions found in the
    gnnLightning.py file.
    """

    def setUp(self):
        """
        Setup the necessary datasets for use in testing later.
        """
        # Set the dtype to be 64 by default
        torch.set_default_dtype(torch.float64)

        # Setup a random generator
        self.rand_gen = torch.Generator().manual_seed(10341885)

        # Initalize the datasets
        path_to_urdf = Path('urdf_files', 'A1', 'a1.urdf').absolute()
        self.path_to_mc_urdf = Path('urdf_files', 'MiniCheetah', 'miniCheetah.urdf').absolute()

        path_to_quad_sdk = Path(
            Path('.').parent, 'datasets', 'QuadSDK-A1Speed1.0').absolute()
        self.path_to_crc_seq = Path(Path('.').parent, 'datasets', 'LinTzuYaun-AR').absolute()

        self.dataset_mlp = QuadSDKDataset_A1Speed1_0(
            path_to_quad_sdk, path_to_urdf, 'package://a1_description/',
            'unitree_ros/robots/a1_description', 'mlp')
        self.dataset_hgnn = QuadSDKDataset_A1Speed1_0(
            path_to_quad_sdk, path_to_urdf, 'package://a1_description/',
            'unitree_ros/robots/a1_description', 'heterogeneous_gnn')
        self.dataset_hgnn_3 = QuadSDKDataset_A1Speed1_0(
            path_to_quad_sdk, path_to_urdf, 'package://a1_description/',
            'unitree_ros/robots/a1_description', 'heterogeneous_gnn', 3)

        self.class_dataset_hgnn_3 = LinTzuYaunDataset_asphalt_road(
            self.path_to_crc_seq, self.path_to_mc_urdf, 'package://yobotics_description/',
            'mini-cheetah-gazebo-urdf/yobo_model/yobotics_description', 
            'heterogeneous_gnn', 3)

        # Put them into an array for easy testing over all of them
        self.models = [self.dataset_mlp, self.dataset_hgnn, self.dataset_hgnn_3]

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

            # Make sure two models were saved
            models = sorted(Path('.', path_to_ckpt_folder).glob(("epoch=*")))
            self.assertEqual(len(models), 2)

            # Predict with the model
            pred, labels = evaluate_model(models[0], test_dataset)

            # Assert the sizes of the results match
            self.assertEqual(pred.shape[0], len(test_dataset.indices))
            self.assertEqual(pred.shape[1], 4)
            self.assertEqual(labels.shape[0], len(test_dataset.indices))
            self.assertEqual(labels.shape[1], 4)

        # Test for classification
        train_dataset, val_dataset, test_dataset = random_split(
            self.class_dataset_hgnn_3, [0.7, 0.2, 0.1], generator=self.rand_gen)
        path_to_ckpt_folder = train_model(train_dataset, val_dataset, test_dataset,
                                            testing_mode=True, disable_logger=True, 
                                            regression=False)

        # Make sure two models were saved
        models = sorted(Path('.', path_to_ckpt_folder).glob(("epoch=*")))
        self.assertEqual(len(models), 2)

        # Predict with the model
        pred, labels = evaluate_model(models[0], test_dataset)

        # Assert the sizes of the results match
        self.assertEqual(pred.shape[0], len(test_dataset.indices))
        self.assertEqual(labels.shape[0], len(test_dataset.indices))

    def test_reshape_functions_for_heterogeneous_gnn(self):
        """
        For the HGNN, test the following functions:
            1. get_foot_node_outputs_gnn()
            2. get_foot_node_labels_gnn()
        Make sure that these methods properly:
            1. Reshapes the output into each graph output
            2. Extracts the foot nodes output and puts them in the correct order
        """

        def test_helper(self, dataset, regression: bool):
            # Get a small trainset so making small batches runs fast
            train_dataset, val_dataset, test_dataset = random_split(dataset, [0.02, 0.5, 0.48], generator=self.rand_gen)
            
            # Make loaders of different batch sizes
            trainLoader_b1: DataLoader = DataLoader(train_dataset, batch_size=1, shuffle=False, num_workers=15)
            trainLoader_b100: DataLoader = DataLoader(train_dataset, batch_size=100, shuffle=False, num_workers=15)

            # Get a dummy batch
            dummy_batch = None
            for batch in trainLoader_b1:
                dummy_batch = batch
                break
            
            # Create a model for running the tests
            model = Heterogeneous_GNN_Lightning(
                hidden_channels=10,
                edge_dim=dummy_batch['base', 'connect',
                                    'joint'].edge_attr.size()[1],
                num_layers=8,
                data_metadata=train_dataset.dataset.get_data_metadata(),
                dummy_batch=dummy_batch,
                optimizer="adam",
                lr=0.003,
                regression=regression)
            
            # Get an example batch for each batch size
            batch_1 = None
            batch_100 = None
            for val in trainLoader_b1:
                batch_1 = val
                break
            for val in trainLoader_b100:
                batch_100 = val
                break

            # Get the raw model output
            raw_out_batch_size_1 = model.model(x_dict=batch_1.x_dict, edge_index_dict=batch_1.edge_index_dict)
            raw_out_batch_size_100 = model.model(x_dict=batch_100.x_dict, edge_index_dict=batch_100.edge_index_dict)

            # Get the reshaping output
            result_1 = get_foot_node_outputs_gnn(raw_out_batch_size_1, batch_1, regression)
            result_100 = get_foot_node_outputs_gnn(raw_out_batch_size_100, batch_100, regression)

            # Test that the kept outputs come from the feet nodes, and are in the correct order
            indices = [0, 1, 2, 3]
            np.testing.assert_array_equal(raw_out_batch_size_1[indices].flatten().unsqueeze(0).detach().numpy(), 
                                        result_1.detach().numpy())
            for i in range(0, int(raw_out_batch_size_100.shape[0] / 4)):
                indices_shifted = [x + 4 * i for x in indices]
                np.testing.assert_array_equal(raw_out_batch_size_100[indices_shifted].flatten().detach().numpy(), 
                                            result_100[i].detach().numpy())

            # Assert that the output of the single batch ends up in the correct spot
            # in the multi-batch output
            np.testing.assert_array_almost_equal(result_1.detach().numpy(), result_100[0].unsqueeze(0).detach().numpy(), 15)

            # Test that the y reshaping holds the same property
            y_1 = get_foot_node_labels_gnn(batch_1)
            y_100 = get_foot_node_labels_gnn(batch_100)
            np.testing.assert_array_equal(y_1.squeeze().numpy(), y_100[0].numpy())

            # TODO: Add checks that the outputs are in the shapes we are expecting

        # Test the regression model
        test_helper(self, self.dataset_hgnn, regression=True)

        # Test the classification model
        test_helper(self, self.class_dataset_hgnn_3, regression=False)


    def test_loss_calculation_for_epoch(self):
        """
        This helper method ensures that the loss calculation after an epoch
        correctly combines the results of multiple batches, and that this
        will work even for two seperate epochs that use the same metric modules.
        """

        def test_regression_loss_helper(base: Base_Lightning, a_size, b_size):
            """
            This helper method makes sure that the losses 
            are calculated correctly for an epoch.
            """

            # Generate two random "predicted" tensors and two random "GT" tensors
            a_pred = torch.rand(a_size)
            a_gt = torch.rand(a_size)
            b_pred = torch.rand(b_size)
            b_gt = torch.rand(b_size)

            # Use the Base lightning methods to calculate the losses for the epoch
            base.calculate_losses_step(a_gt, a_pred)
            base.calculate_losses_step(b_gt, b_pred)
            base.calculate_losses_epoch()

            # Calculate the desired losses
            pred = torch.concat((a_pred, b_pred), 0)
            gt = torch.concat((a_gt, b_gt), 0)
            mse_loss_des = torch.nn.functional.mse_loss(pred, gt)
            rmse_loss_des = torch.sqrt(mse_loss_des)
            l1_loss_des = torch.nn.functional.l1_loss(pred, gt)
            
            # Make sure that they match
            np.testing.assert_array_almost_equal(mse_loss_des, base.mse_loss, 16)
            np.testing.assert_array_almost_equal(rmse_loss_des, base.rmse_loss, 16)
            np.testing.assert_array_almost_equal(l1_loss_des, base.l1_loss, 15)

        def test_classification_loss_helper(base: Base_Lightning, a_size, b_size):
            """
            This helper method makes sure that the classification 
            losses are calculated for an epoch.
            """

            a_pred = torch.nn.functional.normalize(torch.rand((a_size, 8)), p=1, dim=1)
            a_gt = torch.zeros((a_size, 4), dtype=int)
            for i in range(0, a_size):
                a_gt[i] = int(torch.randint(0, 2, (1,)))
            b_pred = torch.nn.functional.normalize(torch.rand((b_size, 8)), p=1, dim=1)
            b_gt = torch.zeros((b_size, 4), dtype=int)
            for i in range(0, b_size):
                b_gt[i] = int(torch.randint(0, 2, (1,)))
            
            # Use the Base lightning methods to calculate the losses for the epoch
            base.calculate_losses_step(a_gt, a_pred)
            base.calculate_losses_step(b_gt, b_pred)
            base.calculate_losses_epoch()

            # Calculate the desired losses
            pred = torch.concat((a_pred, b_pred), 0)
            gt = torch.concat((a_gt, b_gt), 0).long().flatten()
            ce_loss_des = torch.nn.functional.cross_entropy(torch.reshape(pred, (pred.shape[0]*4, 2)), gt)
            
            # Make sure that they match
            np.testing.assert_array_almost_equal(ce_loss_des, base.ce_loss, 7)

        # Define a BaseLightning model
        base = Base_Lightning("adam", 0.003, True)
        
        # Make sure the loss calculation works properly 
        # after we reset the torchmetric module for regression
        test_regression_loss_helper(base, 100, 148)
        base.reset_all_metrics()
        test_regression_loss_helper(base, 3, 250)
        base.reset_all_metrics()
        test_regression_loss_helper(base, 777, 777)
        base.reset_all_metrics()

        # Define another one for classification losses
        base = Base_Lightning("adam", 0.003, False)

        # Make sure the loss calculation works properly 
        # after we reset the torchmetric module for classification
        test_classification_loss_helper(base, 2, 2)
        base.reset_all_metrics()
        test_classification_loss_helper(base, 3, 250)
        base.reset_all_metrics()
        test_classification_loss_helper(base, 777, 777)
        base.reset_all_metrics()

        # TODO: Add Accuracy and F1-Score to this test case

    def test_gnn_classification_output(self):
        """
        Helper function which ensures that the labels and predictions for each individual foot
        are changed into 16 class labels and predictions.
        """

        # Define inputs and desired outputs
        y_pred = torch.tensor([[0.1, 0.9, 0.7, 0.3, 0.2, 0.8, 0.45, 0.55],
                               [0.5, 0.5, 0.5, 0.5, 0.5, 0.5,  0.5,  0.5]], dtype=torch.float64)
        y = torch.tensor([[1, 0, 1, 1],
                          [0, 1, 1, 0]], dtype=torch.int)

        y_pred_new_des = torch.tensor([[1.45, 1.55, 2.05, 2.15, 
                                        1.05, 1.15, 1.65, 1.75,
                                        2.25, 2.35, 2.85, 2.95,
                                        1.85, 1.95, 2.45, 2.55],
                                       [2.0, 2.0, 2.0, 2.0,
                                        2.0, 2.0, 2.0, 2.0,
                                        2.0, 2.0, 2.0, 2.0,
                                        2.0, 2.0, 2.0, 2.0]], dtype=torch.float64)
        y_new_des = torch.tensor([[11],
                                  [6]], dtype=torch.int)

        # Test that the function works properly
        y_new, y_pred_new = gnn_classification_output(y, y_pred)
        np.testing.assert_array_almost_equal(y_pred_new_des.numpy(), y_pred_new.numpy(), 15)
        np.testing.assert_array_almost_equal(y_new_des.numpy(), y_new.numpy(), 15)

if __name__ == "__main__":
    unittest.main()
