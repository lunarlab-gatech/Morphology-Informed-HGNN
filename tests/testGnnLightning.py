import unittest
from pathlib import Path
from grfgnn import QuadSDKDataset_A1Speed1_0, QuadSDKDataset_A1Speed0_5, QuadSDKDataset_A1Speed1_5FlippedOver
from grfgnn.gnnLightning import train_model, evaluate_model, get_foot_node_outputs_gnn, GCN_Lightning, get_foot_node_labels_gnn, Heterogeneous_GNN_Lightning
import torch
from torch.utils.data import random_split
import numpy as np
from torch_geometric.loader import DataLoader
from torch_geometric.data import Batch
from torch_geometric.nn import GCNConv

class TestGnnLighting(unittest.TestCase):
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

    def test_reshape_functions_for_normal_gnn(self):
        """
        For the GNN, test the following functions:
            1. get_foot_node_outputs_gnn()
            2. get_foot_node_labels_gnn()
        Make sure that these methods properly:
            1. Reshapes the output into each graph output
            2. Extracts the foot nodes output and puts them in the correct order
        """

        # Get a small trainset so making small batches runs fast
        train_dataset, val_dataset, test_dataset = random_split(self.dataset_gnn, [0.02, 0.5, 0.48], generator=self.rand_gen)
        
        # Make loaders of different batch sizes
        trainLoader_b1: DataLoader = DataLoader(train_dataset, batch_size=1, shuffle=False, num_workers=15)
        trainLoader_b100: DataLoader = DataLoader(train_dataset, batch_size=100, shuffle=False, num_workers=15)
        
        # Create a model for running the tests
        model = GCN_Lightning(train_dataset[0].x.shape[1], 10, 8, 
                              train_dataset.dataset.get_foot_node_indices_matching_labels())
        
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
        raw_out_batch_size_1 = model.gcn_model(x=batch_1.x,edge_index=batch_1.edge_index)
        raw_out_batch_size_100 = model.gcn_model(x=batch_100.x,edge_index=batch_100.edge_index)

        # Get the reshaping output
        result_1 = get_foot_node_outputs_gnn(raw_out_batch_size_1, batch_1, model.y_indices, "gnn")
        result_100 = get_foot_node_outputs_gnn(raw_out_batch_size_100, batch_100, model.y_indices, "gnn")

        # Test that the kept outputs come from the feet nodes, and are in the correct order
        np.testing.assert_array_equal(raw_out_batch_size_1[model.y_indices].squeeze().detach().numpy(), 
                                      result_1.squeeze().detach().numpy())
        for i in range(0, int(raw_out_batch_size_100.shape[0] / 17)):
            indices_shifted = [x + 17 * i for x in model.y_indices]
            np.testing.assert_array_equal(raw_out_batch_size_100[indices_shifted].squeeze().detach().numpy(), 
                                        result_100[i].squeeze().detach().numpy())

        # Assert that the output of the single batch ends up in the correct spot
        # in the multi-batch output
        np.testing.assert_array_equal(result_1.squeeze().detach().numpy(), result_100[0].detach().numpy())

        # Test that the y reshaping holds the same property
        y_1 = get_foot_node_labels_gnn(batch_1, model.y_indices)
        y_100 = get_foot_node_labels_gnn(batch_100, model.y_indices)
        np.testing.assert_array_equal(y_1.squeeze().numpy(), y_100[0].numpy())

    def test_reshape_functions_for_heterogenous_gnn(self):
        """
        For the HGNN, test the following functions:
            1. get_foot_node_outputs_gnn()
            2. get_foot_node_labels_gnn()
        Make sure that these methods properly:
            1. Reshapes the output into each graph output
            2. Extracts the foot nodes output and puts them in the correct order
        """

        # Get a small trainset so making small batches runs fast
        train_dataset, val_dataset, test_dataset = random_split(self.dataset_hgnn, [0.02, 0.5, 0.48], generator=self.rand_gen)
        
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
            y_indices=train_dataset.dataset.get_foot_node_indices_matching_labels(),
            data_metadata=train_dataset.dataset.get_data_metadata(),
            dummy_batch=dummy_batch)
        
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
        result_1 = get_foot_node_outputs_gnn(raw_out_batch_size_1, batch_1, model.y_indices, "heterogeneous_gnn")
        result_100 = get_foot_node_outputs_gnn(raw_out_batch_size_100, batch_100, model.y_indices, "heterogeneous_gnn")

        # Test that the kept outputs come from the feet nodes, and are in the correct order
        np.testing.assert_array_equal(raw_out_batch_size_1[model.y_indices].squeeze().detach().numpy(), 
                                      result_1.squeeze().detach().numpy())
        for i in range(0, int(raw_out_batch_size_100.shape[0] / 4)):
            indices_shifted = [x + 4 * i for x in model.y_indices]
            np.testing.assert_array_equal(raw_out_batch_size_100[indices_shifted].squeeze().detach().numpy(), 
                                        result_100[i].squeeze().detach().numpy())

        # Assert that the output of the single batch ends up in the correct spot
        # in the multi-batch output
        np.testing.assert_array_almost_equal(result_1.squeeze().detach().numpy(), result_100[0].detach().numpy(), 15)

        # Test that the y reshaping holds the same property
        y_1 = get_foot_node_labels_gnn(batch_1, model.y_indices)
        y_100 = get_foot_node_labels_gnn(batch_100, model.y_indices)
        np.testing.assert_array_equal(y_1.squeeze().numpy(), y_100[0].numpy())

if __name__ == "__main__":
    unittest.main()
