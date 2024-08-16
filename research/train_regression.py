from pathlib import Path
from grfgnn.lightning_py.gnnLightning import train_model
import torch
from grfgnn import QuadSDKDataset_A1Speed0_5, QuadSDKDataset_A1Speed1_0, QuadSDKDataset_A1Speed1_5FlippedOver
from torch.utils.data import Subset

def main():
    # TODO: Do we need to use the same A1 URDF file from QuadSDK?
    path_to_urdf = Path('urdf_files', 'A1', 'a1.urdf').absolute()
    path_to_quad_sdk_05 = Path(
            Path('.').parent, 'datasets', 'QuadSDK-A1Speed0.5').absolute()
    path_to_quad_sdk_1 = Path(
            Path('.').parent, 'datasets', 'QuadSDK-A1Speed1.0').absolute()
    path_to_quad_sdk_15Flipped = Path(
            Path('.').parent, 'datasets', 'QuadSDK-A1Speed1.5FlippedOver').absolute()

    # Define model type
    model_type = 'heterogeneous_gnn'

    # Initalize the datasets
    dataset_05 = QuadSDKDataset_A1Speed0_5(
        path_to_quad_sdk_05, path_to_urdf, 'package://a1_description/',
        'unitree_ros/robots/a1_description', model_type, 5)
    dataset_1 = QuadSDKDataset_A1Speed1_0(
        path_to_quad_sdk_1, path_to_urdf, 'package://a1_description/',
        'unitree_ros/robots/a1_description', model_type, 5)
    dataset_15Flipped = QuadSDKDataset_A1Speed1_5FlippedOver(
        path_to_quad_sdk_15Flipped, path_to_urdf, 'package://a1_description/',
        'unitree_ros/robots/a1_description', model_type, 5)


    # Split the data into training, validation, and testing sets
    rand_gen = torch.Generator().manual_seed(10341885)
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
        dataset_1, [0.8, 0.1, 0.1], generator=rand_gen)

    # Train the model
    train_model(train_dataset, val_dataset, test_dataset, num_layers=9, hidden_size=32, disable_logger=True)

if __name__ == '__main__':
     main()