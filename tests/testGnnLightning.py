import unittest
from pathlib import Path
import torchmetrics.classification
from mi_hgnn import QuadSDKDataset_A1Speed1_0_DEPRECATED
import mi_hgnn.datasets_py.LinTzuYaunDataset as linData
from mi_hgnn.datasets_py.LinTzuYaunDataset import LinTzuYaunDataset_asphalt_road
from mi_hgnn.lightning_py.gnnLightning import train_model, evaluate_model, Heterogeneous_GNN_Lightning, Base_Lightning
from mi_hgnn.visualization import visualize_model_outputs_regression
from mi_hgnn.datasets_py.quadSDKDataset import *
import torch
import lightning as L
from torch.utils.data import random_split
import numpy as np
import torchmetrics
from torch_geometric.loader import DataLoader
import os

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

        # Initalize the simple datasets
        path_to_urdf = Path('urdf_files', 'A1', 'a1.urdf').absolute()
        self.path_to_mc_urdf = Path('urdf_files', 'MiniCheetah', 'miniCheetah.urdf').absolute()

        path_to_quad_sdk = Path(
            Path('.').parent, 'datasets', 'QuadSDK-A1Speed1.0').absolute()
        self.path_to_crc_seq = Path(Path('.').parent, 'datasets', 'LinTzuYaun-AR').absolute()

        self.dataset_mlp = QuadSDKDataset_A1Speed1_0_DEPRECATED(
            path_to_quad_sdk, path_to_urdf, 'package://a1_description/',
            'unitree_ros/robots/a1_description', 'mlp')
        self.dataset_hgnn = QuadSDKDataset_A1Speed1_0_DEPRECATED(
            path_to_quad_sdk, path_to_urdf, 'package://a1_description/',
            'unitree_ros/robots/a1_description', 'heterogeneous_gnn')
        self.dataset_hgnn_3 = QuadSDKDataset_A1Speed1_0_DEPRECATED(
            path_to_quad_sdk, path_to_urdf, 'package://a1_description/',
            'unitree_ros/robots/a1_description', 'heterogeneous_gnn', 3)

        self.class_dataset_mlp = LinTzuYaunDataset_asphalt_road(
            self.path_to_crc_seq, self.path_to_mc_urdf, 'package://yobotics_description/',
            'mini-cheetah-gazebo-urdf/yobo_model/yobotics_description', 
            'mlp', 1)
        self.class_dataset_hgnn = LinTzuYaunDataset_asphalt_road(
            self.path_to_crc_seq, self.path_to_mc_urdf, 'package://yobotics_description/',
            'mini-cheetah-gazebo-urdf/yobo_model/yobotics_description', 
            'heterogeneous_gnn', 1)
        self.class_dataset_hgnn_3 = LinTzuYaunDataset_asphalt_road(
            self.path_to_crc_seq, self.path_to_mc_urdf, 'package://yobotics_description/',
            'mini-cheetah-gazebo-urdf/yobo_model/yobotics_description', 
            'heterogeneous_gnn', 3)
        
        # Put them into an array for easy testing over all of them
        self.models = [self.dataset_mlp, self.dataset_hgnn, self.dataset_hgnn_3]
        self.class_models = [self.class_dataset_hgnn, self.class_dataset_hgnn_3]
        
        # Create the test dataset for classification
        history_length = 150
        model_type = 'heterogeneous_gnn'
        air_jumping_gait = linData.LinTzuYaunDataset_air_jumping_gait(
            Path(Path('.').parent, 'datasets', 'LinTzuYaun-AJG').absolute(), self.path_to_mc_urdf, 'package://yobotics_description/', 'mini-cheetah-gazebo-urdf/yobo_model/yobotics_description', model_type, history_length, normalize=True)
        concrete_pronking = linData.LinTzuYaunDataset_concrete_pronking(
            Path(Path('.').parent, 'datasets', 'LinTzuYaun-CP').absolute(), self.path_to_mc_urdf, 'package://yobotics_description/', 'mini-cheetah-gazebo-urdf/yobo_model/yobotics_description', model_type, history_length, normalize=True)
        concrete_right_circle = linData.LinTzuYaunDataset_concrete_right_circle(
            Path(Path('.').parent, 'datasets', 'LinTzuYaun-CRC').absolute(), self.path_to_mc_urdf, 'package://yobotics_description/', 'mini-cheetah-gazebo-urdf/yobo_model/yobotics_description', model_type, history_length, normalize=True)
        forest = linData.LinTzuYaunDataset_forest(
            Path(Path('.').parent, 'datasets', 'LinTzuYaun-F').absolute(), self.path_to_mc_urdf, 'package://yobotics_description/', 'mini-cheetah-gazebo-urdf/yobo_model/yobotics_description', model_type, history_length, normalize=True)
        small_pebble = linData.LinTzuYaunDataset_small_pebble(
            Path(Path('.').parent, 'datasets', 'LinTzuYaun-SP').absolute(), self.path_to_mc_urdf, 'package://yobotics_description/', 'mini-cheetah-gazebo-urdf/yobo_model/yobotics_description', model_type, history_length, normalize=True)
        test_dataset = torch.utils.data.ConcatDataset([air_jumping_gait, concrete_pronking, concrete_right_circle, forest, small_pebble])
        self.test_dataset = torch.utils.data.Subset(test_dataset, np.arange(0, test_dataset.__len__()))

    def test_methods_run_without_crashing(self):
        """
        Make sure that the train model function runs and
        finishes successfully. Also, test that we can evaluate
        each model as well without a crash.
        """

        # For each regression model
        for i, model in enumerate(self.models):
            train_dataset, val_dataset, test_dataset = random_split(
                model, [0.7, 0.2, 0.1], generator=self.rand_gen)
            path_to_ckpt_folder = train_model(train_dataset, val_dataset, test_dataset,
                                              normalize=False,
                                              testing_mode=True, disable_logger=True,
                                              epochs=2, seed=1919)

            # Make sure three models were saved (2 for top, 1 for last)
            models = sorted(Path('.', path_to_ckpt_folder).glob(("epoch=*")))
            self.assertEqual(len(models), 3)
            
            try:
                pred, labels, mse_loss, rmse_loss, l1_loss = evaluate_model(models[0], test_dataset)

                # Assert the sizes of the results match
                self.assertEqual(pred.shape[0], test_dataset.__len__())
                self.assertEqual(pred.shape[1], 4)
                self.assertEqual(labels.shape[0], test_dataset.__len__())
                self.assertEqual(labels.shape[1], 4)

                # Try and visualize with the model
                visualize_model_outputs_regression(pred, labels, no_show=True)

            except Exception as e:
                for path in models:
                    Path.unlink(path, missing_ok=False)
                raise e
            
            for path in models:
                Path.unlink(path, missing_ok=False)

        # For each classification model
        # TODO: Test for MLP
        for i, model in enumerate(self.class_models):
            # Test for classification
            train_dataset, val_dataset, test_dataset = random_split(
                model, [0.7, 0.2, 0.1], generator=self.rand_gen)
            path_to_ckpt_folder = train_model(train_dataset, val_dataset, test_dataset,
                                              normalize=False,
                                                testing_mode=True, disable_logger=True, 
                                                regression=False, epochs=2, seed=1919)

            # Make sure three models were saved (2 for top, 1 for last)
            models = sorted(Path('.', path_to_ckpt_folder).glob(("epoch=*")))
            self.assertEqual(len(models), 3)

            try:
                # Evaluate the model
                pred, labels, acc, f1_leg_0, f1_leg_1, f1_leg_2, f1_leg_3, f1_avg_legs = evaluate_model(models[0], test_dataset)

                # Assert the sizes of the results match
                self.assertEqual(pred.shape[0], test_dataset.__len__())
                self.assertEqual(labels.shape[0], test_dataset.__len__())
                
            except Exception as e:
                for path in models:
                    Path.unlink(path, missing_ok=False)
                raise e
            
            for path in models:
                Path.unlink(path, missing_ok=False)

    @unittest.skipIf(os.environ.get('MIHGNN_UNITTEST_SKIP_GITHUB_ACTION_CRASHERS') == "True", "Skipping tests that crash GitHub Actions")
    def test_evaluate_model(self):
        """
        Test that the evaluation method properly calculates
        the same values logged during the training process.
        """

        # ======================== Regression with MLP/Dynamic ========================
        # Create the test dataset
        path_to_urdf = Path('urdf_files', 'Go2-Quad', 'go2.urdf').absolute()
        model_type = 'mlp'
        history_length = 1
        normalize = False
        dataset_flat_0_5_50 = QuadSDKDataset_Go2_Flat_Speed0_5_Mu_50(Path(Path('.').parent, 'datasets', 'QuadSDK-Go2-Flat-0.5-Mu50').absolute(), 
                path_to_urdf, 'package://go2_description/', '', model_type, history_length, normalize=normalize)
        predict_dataset = torch.utils.data.Subset(dataset_flat_0_5_50, np.arange(0, 2))

        # Use the predict methods
        pred, y, mse_loss, rmse_loss, l1_loss = evaluate_model(None, predict_dataset, enable_testing_mode=True)

        # Make sure that they're the values that we expect
        x_0, y_0_des = predict_dataset.dataset.get(0)
        x_1, y_1_des = predict_dataset.dataset.get(1)
        np.testing.assert_array_equal(y[0].numpy(), y_0_des)
        np.testing.assert_array_equal(y[1].numpy(), y_1_des)
        np.testing.assert_array_equal(pred[0].numpy(), y_0_des * 3)
        np.testing.assert_array_equal(pred[1].numpy(), y_1_des * 3)

        # Make sure the loss matches what we expect
        mse_loss_des = torch.nn.functional.mse_loss(pred, y)
        rmse_loss_des = torch.sqrt(mse_loss_des)
        l1_loss_des = torch.nn.functional.l1_loss(pred, y)

        np.testing.assert_equal(mse_loss.item(), mse_loss_des.item())
        np.testing.assert_equal(rmse_loss.item(), rmse_loss_des.item())
        np.testing.assert_equal(l1_loss.item(), l1_loss_des.item())

        # ======================== Regression with HGNN ========================
        # Create the predict dataset
        path_to_urdf = Path('urdf_files', 'Go2-Quad', 'go2.urdf').absolute()
        model_type = 'heterogeneous_gnn'
        history_length = 1
        normalize = False
        regression_hgnn_dataset = QuadSDKDataset_Go2_Flat_Speed0_5_Mu_50(Path(Path('.').parent, 'datasets', 'QuadSDK-Go2-Flat-0.5-Mu50').absolute(), 
            path_to_urdf, 'package://go2_description/', '', model_type, history_length, normalize=normalize)
        regression_hgnn_dataset = torch.utils.data.Subset(regression_hgnn_dataset, np.arange(0, 20))

        # Use the predict methods
        path_to_checkpoint = Path('tests', 'test_models', 'epoch=48-val_MSE_loss=6.33834.ckpt').absolute()
        pred, y, mse_loss, rmse_loss, l1_loss = evaluate_model(str(path_to_checkpoint), regression_hgnn_dataset)

        # Make sure that they're the values that we expect
        y_0_des = regression_hgnn_dataset.dataset.get(0).y
        y_1_des = regression_hgnn_dataset.dataset.get(19).y
        np.testing.assert_array_equal(y[0].numpy(), y_0_des)
        np.testing.assert_array_equal(y[19].numpy(), y_1_des)

        # Make sure the loss matches what we expect
        np.testing.assert_almost_equal(mse_loss.item(), 6.33834, 5)
        np.testing.assert_almost_equal(rmse_loss.item(), 2.51761, 5)
        np.testing.assert_almost_equal(l1_loss.item(), 2.31058, 5)

        # ======================== Classification with HGNN ========================
        # Evaluate with model
        path_to_checkpoint = Path('tests', 'test_models', 'epoch=10-val_CE_loss=0.30258.ckpt').absolute()
        pred, labels, acc, f1_leg_0, f1_leg_1, f1_leg_2, f1_leg_3, f1_avg_legs = evaluate_model(str(path_to_checkpoint), self.test_dataset)

        # Assert that the evaluated metrics match what we calculated using
        # the test methods during training
        np.testing.assert_almost_equal(acc.item(), 0.89581, 4)
        np.testing.assert_almost_equal(f1_leg_0.item(), 0.94857, 4)
        np.testing.assert_almost_equal(f1_leg_1.item(), 0.95383, 4)
        np.testing.assert_almost_equal(f1_leg_2.item(), 0.94795, 4)
        np.testing.assert_almost_equal(f1_leg_3.item(), 0.9464, 4)
        np.testing.assert_almost_equal(f1_avg_legs.item(), 0.94918, 4)

        # # Assert the pred and labels are correct
        metric_acc = torchmetrics.Accuracy(task="multiclass", num_classes=16)
        calculated_acc = metric_acc(labels, pred)
        np.testing.assert_equal(acc.item(), calculated_acc.item())

        # Make sure that they're the values that we expect
        y_0_des = self.test_dataset.dataset.__getitem__(0).y
        y_1_des = self.test_dataset.dataset.__getitem__(1000).y
        np.testing.assert_array_equal(labels[0].numpy(), y_0_des)
        np.testing.assert_array_equal(labels[1000].numpy(), y_1_des)

    def test_MIHGNN_model_output_assumption(self):
        """
        The code as written assumes that the MIHGNN model output follows
        the corresponding convention. Given the output is a tensor
        of shape (batch_size * 4, out_channels), we assume that:
        - Every four sequential values corresponds to the outputs 
          of the feet nodes in a particular graph, in the order
          of the URDF file.
        - Treating each group of four values as the output of a graph,
          the order of the graphs is the same as the order of the input
          graphs.

        This method verifies these assumptions. 
        """
        batch_size = 100
        trainLoader: DataLoader = DataLoader(self.class_dataset_hgnn, batch_size=batch_size,
                                         shuffle=False, num_workers=1)
        
        dummy_batch = None
        for batch in trainLoader:
            dummy_batch = batch
            break

        # Test the the HeteroDataBatch gets things in the order we assume
        des_order = []
        for i in range(0, batch_size):
            for j in range(0, 4):
                des_order.append(i)
        des_order = np.array(des_order)
        np.testing.assert_array_equal(des_order, batch['foot'].batch.numpy())

        for i in range(0, batch_size):
            np.testing.assert_array_equal(batch['foot'].x[i*4:(i+1)*4,:], self.class_dataset_hgnn.get(i)['foot'].x)
            np.testing.assert_array_equal(batch.y[i*4:(i+1)*4], self.class_dataset_hgnn.get(i).y)

        # Assert that when we get x_dict, the foot values stay in the same order
        np.testing.assert_array_equal(batch.x_dict['foot'], batch['foot'].x)

        # Therefore, the MIHGNN model outputs are also in the same order as above.

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

            a_pred = torch.rand((a_size, 8))
            a_gt = torch.zeros((a_size, 4), dtype=int)
            for i in range(0, a_size):
                a_gt[i] = int(torch.randint(0, 2, (1,)))
            b_pred = torch.rand((b_size, 8))
            b_gt = torch.zeros((b_size, 4), dtype=int)
            for i in range(0, b_size):
                b_gt[i] = int(torch.randint(0, 2, (1,)))
            
            # Use the Base lightning methods to calculate the losses for the epoch
            base.calculate_losses_step(a_gt, a_pred)
            base.calculate_losses_step(b_gt, b_pred)
            base.calculate_losses_epoch()

            # Calculate the desired losses
            pred = torch.concat((a_pred, b_pred), 0)
            pred_reshaped = torch.reshape(torch.concat((a_pred, b_pred), 0), (pred.shape[0] * 4, 2))
            gt = torch.concat((a_gt, b_gt), 0)
            gt_reshaped = gt.long().flatten()
            loss_fn = torch.nn.CrossEntropyLoss(reduction='mean')
            ce_loss_des = loss_fn(pred_reshaped, gt_reshaped)

            y_pred_per_foot, y_pred_per_foot_prob, y_pred_per_foot_prob_only_1 = \
                    base.classification_calculate_useful_values(pred, pred.shape[0])
            y_pred_16, y_16 = base.classification_conversion_16_class(y_pred_per_foot_prob_only_1, gt)
            metric_acc = torchmetrics.Accuracy(task="multiclass", num_classes=16)
            acc_des = metric_acc(torch.argmax(y_pred_16, dim=1), y_16.squeeze())

            y_pred_2 = torch.reshape(torch.argmax(y_pred_per_foot_prob, dim=1), (pred.shape[0], 4))
            metric_f1 =torchmetrics.classification.BinaryF1Score()
            f1_leg0 = metric_f1(y_pred_2[:,0], gt[:,0])
            f1_leg1 = metric_f1(y_pred_2[:,1], gt[:,1])
            f1_leg2 = metric_f1(y_pred_2[:,2], gt[:,2])
            f1_leg3 = metric_f1(y_pred_2[:,3], gt[:,3])

            # Make sure that they match
            np.testing.assert_almost_equal(ce_loss_des.item(), base.ce_loss.item(), 7)
            np.testing.assert_almost_equal(acc_des.item(), base.acc.item(), 16)
            np.testing.assert_almost_equal(f1_leg0.item(), base.f1_leg0.item(), 16)
            np.testing.assert_almost_equal(f1_leg1.item(), base.f1_leg1.item(), 16)
            np.testing.assert_almost_equal(f1_leg2.item(), base.f1_leg2.item(), 16)
            np.testing.assert_almost_equal(f1_leg3.item(), base.f1_leg3.item(), 16)

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

    def test_classification_conversion_16_class(self):
        """
        Helper function which ensures that the labels and predictions for each individual foot
        are changed into 16 class labels and predictions.
        """

        # Define inputs and desired outputs
        y_pred = torch.tensor([[0.9, 0.3, 0.8, 0.55],
                               [0.5, 0.5, 0.5,  0.5]], dtype=torch.float64)
        y = torch.tensor([[1, 0, 1, 1],
                          [0, 1, 1, 0]], dtype=torch.int)

        y_pred_new_des = torch.tensor([[0.0063, 0.0077, 0.0252, 0.0308, 
                                        0.0027, 0.0033, 0.0108, 0.0132,
                                        0.0567, 0.0693, 0.2268, 0.2772,
                                        0.0243, 0.0297, 0.0972, 0.1188],
                                       [0.0625, 0.0625, 0.0625, 0.0625,
                                        0.0625, 0.0625, 0.0625, 0.0625,
                                        0.0625, 0.0625, 0.0625, 0.0625,
                                        0.0625, 0.0625, 0.0625, 0.0625]], dtype=torch.float64)
        y_new_des = torch.tensor([[11],
                                  [6]], dtype=torch.int)

        # Test that the function works properly
        y_pred_new, y_new= Base_Lightning.classification_conversion_16_class(None, y_pred, y)
        np.testing.assert_array_almost_equal(y_pred_new_des.numpy(), y_pred_new.numpy(), 15)
        np.testing.assert_array_almost_equal(y_new_des.numpy(), y_new.numpy(), 15)

    def test_regression_metric_calculations(self):
        """
        Make sure that our regression metric 
        calculations are as we expect.
        """

        # Set our own y_pred and y values
        y_pred = torch.tensor([[1, 1, 14, -50],
                               [13, 13, 3, 12],
                               [-101, 12, -13, -31]], dtype=torch.float64)
        y = torch.tensor([[12, 13, 114, 0],
                          [1, 1, 0, 1],
                          [-100, 11, -10, -3]], dtype=torch.int)

        # Calculate classification metrics
        base = Base_Lightning('adam', 0.0001, True)
        base.calculate_losses_step(y, y_pred)

        # Calculate desired values
        des_mse_loss = torch.mean(torch.square(y_pred.flatten() - y.flatten()))
        des_rmse_loss = torch.sqrt(torch.mean(torch.square(y_pred.flatten() - y.flatten())))
        des_l1_loss = torch.mean(torch.abs(y_pred.flatten() - y.flatten()))

        # Make sure they match what we expect
        np.testing.assert_equal(base.mse_loss.item(), des_mse_loss.item())
        np.testing.assert_equal(base.rmse_loss.item(), des_rmse_loss.item())
        np.testing.assert_equal(base.l1_loss.item(), des_l1_loss.item())

    def test_classification_metric_calculations(self):
        """
        This incredibly important test makes sure that our metric calculations are correct
        for classification.
        """

        # Setup a dummy dataloader so we can get a batch
        trainLoader: DataLoader = DataLoader(self.class_dataset_hgnn_3,
                                    batch_size=4, shuffle=True, num_workers=1)
        
        # Extract a batch
        batch = None
        for batch in trainLoader:
            batch = batch
            break

        # Set our own y_pred and y values
        y_pred = torch.tensor([[0.1, 11, 100, 19, 0.12, 0.14, 15, 24.45],
                               [15, 11, 19, 19, 0.9898, 0.14, -10000, 24.45],
                               [0.1, 13, 100, 19, 0.12, -10, 15, -24.45],
                               [15, 11, 200, 19, 0.9898, 0.14, -10000, 44.45],
                               [-0.1, 11, 100, 19, 0.12, 0.14, 15, 24.45],
                               [-15, 11, 19, 19, -0.9898, 0.14, -10000, 24.45],
                               [-0.1, 13, 100, 19, 0.12, -10, 15, -24.45],
                               [-15, 11, 200, 19, -0.9898, 0.14, -10000, 44.45]], dtype=torch.float64)
        y = torch.tensor([[1, 1, 1, 1],
                          [1, 1, 0, 1],
                          [0, 1, 1, 0],
                          [0, 1, 0, 0],
                          [0, 0, 1, 1],
                          [1, 1, 1, 0],
                          [1, 0, 0, 0],
                          [1, 0, 0, 1]], dtype=torch.int)

        # Calculate classification metrics
        base = Base_Lightning('adam', 0.0001, False)
        base.calculate_losses_step(y, y_pred)

        # Make sure our CE loss matches what we expect
        y_pred_per_leg_log_soft = torch.nn.functional.log_softmax(torch.reshape(y_pred, (y_pred.shape[0] * 4, 2)), dim=1,  dtype=torch.float64)
        y_per_leg = torch.reshape(y, (y.shape[0] * 4, 1))
        des_loss = 0
        for i in range(0, y_pred_per_leg_log_soft.shape[0]):
            des_loss -= y_pred_per_leg_log_soft[i,y_per_leg[i]]
        des_loss = des_loss / y_pred_per_leg_log_soft.shape[0]
        np.testing.assert_almost_equal(base.ce_loss.item(), des_loss.item(), 5)
    
        # Other metric Regression Tests from MorphoSymm-Replication
        self.assertEqual(base.acc.item(), 0.125)
        self.assertEqual(base.f1_leg0.item(), 0.7272727272727272)
        self.assertEqual(base.f1_leg1.item(), 0.0)
        self.assertEqual(base.f1_leg2.item(), 0.75)
        self.assertEqual(base.f1_leg3.item(), 0.8)

if __name__ == "__main__":
    unittest.main()
