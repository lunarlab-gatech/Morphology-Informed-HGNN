from pathlib import Path
from mi_hgnn.lightning_py.gnnLightning import evaluate_model
import torch
import mi_hgnn.datasets_py.LinTzuYaunDataset as linData
import numpy as np


def main():
    # ================================= CHANGE THIS ====================================
    path_to_checkpoint = None # Path to specific checkpoint file
    # ==================================================================================

    # Check that the user filled in the necessary parameters
    if path_to_checkpoint is None:
        raise ValueError("Please provide a checkpoint path by editing this file!")

    # Set parameters
    history_length = 150
    model_type = 'heterogeneous_gnn'
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
    test_dataset = torch.utils.data.ConcatDataset([air_jumping_gait, concrete_pronking, concrete_right_circle, forest, small_pebble])

    # Convert them to subsets
    test_dataset = torch.utils.data.Subset(test_dataset, np.arange(0, test_dataset.__len__()))

    # Evaluate with model
    pred, labels, acc, f1_leg_0, f1_leg_1, f1_leg_2, f1_leg_3, f1_avg_legs = evaluate_model(path_to_checkpoint, test_dataset)

    # Print the results
    print("Model Accuracy: ", acc)
    print("F1-Score Leg 0: ", f1_leg_0)
    print("F1-Score Leg 1: ", f1_leg_1)
    print("F1-Score Leg 2: ", f1_leg_2)
    print("F1-Score Leg 3: ", f1_leg_3)
    print("F1-Score Legs Avg: ", f1_avg_legs)

if __name__ == "__main__":
    main()
