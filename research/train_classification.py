import mi_hgnn.datasets_py.LinTzuYaunDataset as linData
from pathlib import Path
from torch_geometric.loader import DataLoader
import numpy as np
import torch
from mi_hgnn.lightning_py.gnnLightning import train_model

def main():
    """
    Duplicate the experiment found in Section VI-B of "On discrete symmetries 
    of robotics systems: A group-theoretic and data-driven analysis", but training
    on our HGNN instead.
    """

    # ================================= CHANGE THESE ===================================
    model_type = 'heterogeneous_gnn'
    num_layers = 4
    hidden_size = 128
    # ==================================================================================

    # Set model parameters (so they all match)
    history_length = 150
    normalize = True

    # Initialize the Training/Validation datasets
    path_to_urdf = Path('urdf_files', 'MiniCheetah', 'miniCheetah.urdf').absolute()
    air_walking_gait = linData.LinTzuYaunDataset_air_walking_gait(
        Path(Path('.').parent, 'datasets', 'LinTzuYaun-AWG').absolute(), path_to_urdf, 'package://yobotics_description/', 'mini-cheetah-gazebo-urdf/yobo_model/yobotics_description', model_type, history_length, normalize=normalize)
    concrete_difficult_slippery = linData.LinTzuYaunDataset_concrete_difficult_slippery(
        Path(Path('.').parent, 'datasets', 'LinTzuYaun-CDS').absolute(), path_to_urdf, 'package://yobotics_description/', 'mini-cheetah-gazebo-urdf/yobo_model/yobotics_description', model_type, history_length, normalize=normalize)
    concrete_left_circle = linData.LinTzuYaunDataset_concrete_left_circle(
        Path(Path('.').parent, 'datasets', 'LinTzuYaun-CLC').absolute(), path_to_urdf, 'package://yobotics_description/', 'mini-cheetah-gazebo-urdf/yobo_model/yobotics_description', model_type, history_length, normalize=normalize)
    middle_pebble = linData.LinTzuYaunDataset_middle_pebble(
        Path(Path('.').parent, 'datasets', 'LinTzuYaun-MP').absolute(), path_to_urdf, 'package://yobotics_description/', 'mini-cheetah-gazebo-urdf/yobo_model/yobotics_description', model_type, history_length, normalize=normalize)
    rock_road = linData.LinTzuYaunDataset_rock_road(
        Path(Path('.').parent, 'datasets', 'LinTzuYaun-RR').absolute(), path_to_urdf, 'package://yobotics_description/', 'mini-cheetah-gazebo-urdf/yobo_model/yobotics_description', model_type, history_length, normalize=normalize)
    asphalt_road = linData.LinTzuYaunDataset_asphalt_road(
        Path(Path('.').parent, 'datasets', 'LinTzuYaun-AR').absolute(), path_to_urdf, 'package://yobotics_description/', 'mini-cheetah-gazebo-urdf/yobo_model/yobotics_description', model_type, history_length, normalize=normalize)
    concrete_galloping = linData.LinTzuYaunDataset_concrete_galloping(
        Path(Path('.').parent, 'datasets', 'LinTzuYaun-CG').absolute(), path_to_urdf, 'package://yobotics_description/', 'mini-cheetah-gazebo-urdf/yobo_model/yobotics_description', model_type, history_length, normalize=normalize)
    grass = linData.LinTzuYaunDataset_grass(
        Path(Path('.').parent, 'datasets', 'LinTzuYaun-G').absolute(), path_to_urdf, 'package://yobotics_description/', 'mini-cheetah-gazebo-urdf/yobo_model/yobotics_description', model_type, history_length, normalize=normalize)
    old_asphalt_road = linData.LinTzuYaunDataset_old_asphalt_road(
        Path(Path('.').parent, 'datasets', 'LinTzuYaun-OAR').absolute(), path_to_urdf, 'package://yobotics_description/', 'mini-cheetah-gazebo-urdf/yobo_model/yobotics_description', model_type, history_length, normalize=normalize)
    sidewalk = linData.LinTzuYaunDataset_sidewalk(
        Path(Path('.').parent, 'datasets', 'LinTzuYaun-S').absolute(), path_to_urdf, 'package://yobotics_description/', 'mini-cheetah-gazebo-urdf/yobo_model/yobotics_description', model_type, history_length, normalize=normalize)
    train_val_datasets = [air_walking_gait, concrete_difficult_slippery, concrete_left_circle, middle_pebble, rock_road, asphalt_road, concrete_galloping, grass, old_asphalt_road, sidewalk]

    train_subsets = []
    val_subsets = []
    for dataset in train_val_datasets:
        split_index = int(dataset.len() * 0.85)
        train_subsets.append(torch.utils.data.Subset(dataset, np.arange(0, split_index)))
        val_subsets.append(torch.utils.data.Subset(dataset, np.arange(split_index, dataset.len())))
    train_dataset = torch.utils.data.ConcatDataset(train_subsets)
    val_dataset = torch.utils.data.ConcatDataset(val_subsets)

    # Initialize the Testing datasets
    air_jumping_gait = linData.LinTzuYaunDataset_air_jumping_gait(
        Path(Path('.').parent, 'datasets', 'LinTzuYaun-AJG').absolute(), path_to_urdf, 'package://yobotics_description/', 'mini-cheetah-gazebo-urdf/yobo_model/yobotics_description', model_type, history_length, normalize=normalize)
    concrete_pronking = linData.LinTzuYaunDataset_concrete_pronking(
        Path(Path('.').parent, 'datasets', 'LinTzuYaun-CP').absolute(), path_to_urdf, 'package://yobotics_description/', 'mini-cheetah-gazebo-urdf/yobo_model/yobotics_description', model_type, history_length, normalize=normalize)
    concrete_right_circle = linData.LinTzuYaunDataset_concrete_right_circle(
        Path(Path('.').parent, 'datasets', 'LinTzuYaun-CRC').absolute(), path_to_urdf, 'package://yobotics_description/', 'mini-cheetah-gazebo-urdf/yobo_model/yobotics_description', model_type, history_length, normalize=normalize)
    forest = linData.LinTzuYaunDataset_forest(
        Path(Path('.').parent, 'datasets', 'LinTzuYaun-F').absolute(), path_to_urdf, 'package://yobotics_description/', 'mini-cheetah-gazebo-urdf/yobo_model/yobotics_description', model_type, history_length, normalize=normalize)
    small_pebble = linData.LinTzuYaunDataset_small_pebble(
        Path(Path('.').parent, 'datasets', 'LinTzuYaun-SP').absolute(), path_to_urdf, 'package://yobotics_description/', 'mini-cheetah-gazebo-urdf/yobo_model/yobotics_description', model_type, history_length, normalize=normalize)
    test_dataset = torch.utils.data.ConcatDataset([air_jumping_gait, concrete_pronking, concrete_right_circle, forest, small_pebble])
    
    # Convert them to subsets
    train_dataset = torch.utils.data.Subset(train_dataset, np.arange(0, train_dataset.__len__()))
    val_dataset = torch.utils.data.Subset(val_dataset, np.arange(0, val_dataset.__len__()))
    test_dataset = torch.utils.data.Subset(test_dataset, np.arange(0, test_dataset.__len__()))

    # Train the model
    train_model(train_dataset, val_dataset, test_dataset, normalize, num_layers=num_layers, hidden_size=hidden_size, 
                logger_project_name="grfgnn-class-abalation", batch_size=30, regression=False, lr=0.0001, epochs=30)

if __name__ == "__main__":
    main()