from pathlib import Path
from grfgnn.lightning_py.gnnLightning import evaluate_model
from grfgnn import QuadSDKDataset_A1Speed0_5, QuadSDKDataset_A1Speed1_0, QuadSDKDataset_A1Speed1_5FlippedOver
from grfgnn.visualization import visualize_model_outputs_classification
import torch
from torch.utils.data import Subset
import grfgnn.datasets_py.LinTzuYaunDataset as linData
import numpy as np


def main():
    # Give path to checkpoint
    # ================================= FILL THESE IN ===================================
    path_to_checkpoint = "/home/dbutterfield3/Research/state-estimation-gnn/models/lunar-yogurt-58/epoch=20-val_CE_loss=0.48732.ckpt"
    model_type = 'mlp'
    history_length = 150
    
    # ===================================================================================
    path_to_urdf = Path('urdf_files', 'MiniCheetah', 'miniCheetah.urdf').absolute()

    # Initialize the Testing datasets
    air_jumping_gait = linData.LinTzuYaunDataset_air_jumping_gait(
        Path(Path('.').parent, 'datasets', 'LinTzuYaun-AJG').absolute(), path_to_urdf, 'package://yobotics_description/', 'mini-cheetah-gazebo-urdf/yobo_model/yobotics_description', model_type, history_length, normalize=True)
    concrete_pronking = linData.LinTzuYaunDataset_concrete_pronking(
        Path(Path('.').parent, 'datasets', 'LinTzuYaun-CP').absolute(), path_to_urdf, 'package://yobotics_description/', 'mini-cheetah-gazebo-urdf/yobo_model/yobotics_description', model_type, history_length, normalize=True)
    concrete_right_circle = linData.LinTzuYaunDataset_concrete_right_circle(
        Path(Path('.').parent, 'datasets', 'LinTzuYaun-CRC').absolute(), path_to_urdf, 'package://yobotics_description/', 'mini-cheetah-gazebo-urdf/yobo_model/yobotics_description', model_type, history_length, normalize=True)
    forest = linData.LinTzuYaunDataset_forest(
        Path(Path('.').parent, 'datasets', 'LinTzuYaun-F').absolute(), path_to_urdf, 'package://yobotics_description/', 'mini-cheetah-gazebo-urdf/yobo_model/yobotics_description', model_type, history_length, normalize=True)
    small_pebble = linData.LinTzuYaunDataset_small_pebble(
        Path(Path('.').parent, 'datasets', 'LinTzuYaun-SP').absolute(), path_to_urdf, 'package://yobotics_description/', 'mini-cheetah-gazebo-urdf/yobo_model/yobotics_description', model_type, history_length, normalize=True)
    #test_dataset = torch.utils.data.ConcatDataset([air_jumping_gait, concrete_pronking, concrete_right_circle, forest, small_pebble])
    test_dataset = torch.utils.data.ConcatDataset([concrete_pronking])

    # Convert them to subsets
    test_dataset = torch.utils.data.Subset(test_dataset, np.arange(0, test_dataset.__len__()))

    # ================================= FILL THESE IN ===================================
    num_entries_to_visualize = 10000  # Ex: 2000
    # ===================================================================================
    pred, labels = evaluate_model(path_to_checkpoint, test_dataset, num_entries_to_visualize)
    visualize_model_outputs_classification(pred, labels, str(path_to_checkpoint) + ".pdf", 100)

if __name__ == "__main__":
    main()
