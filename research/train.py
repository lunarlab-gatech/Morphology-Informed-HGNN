from pathlib import Path
from grfgnn.gnnLightning import train_model, split_dataset
import torch
from grfgnn.datasets_deprecated import CerberusStreetDataset, CerberusTrackDataset, Go1SimulatedDataset
from grfgnn.datasets import QuadSDKDataset

def train_model_quad_sdk(path_to_urdf, path_to_quad_sdk):
    model_type = 'heterogeneous_gnn'

    # Initalize the dataset
    a1_sim_dataset = QuadSDKDataset(
        path_to_quad_sdk, path_to_urdf, 'package://a1_description/',
        'unitree_ros/robots/a1_description', model_type)

    # Split the data into training, validation, and testing sets
    rand_gen = torch.Generator().manual_seed(10341885)
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
        a1_sim_dataset, [0.7, 0.2, 0.1], generator=rand_gen)

    # Train the model
    train_model(train_dataset, val_dataset, test_dataset)

def main():
    # TODO: Do we need to use the same A1 URDF file from QuadSDK?
    path_to_urdf = Path('urdf_files', 'A1', 'a1.urdf').absolute()
    path_to_quad_sdk = Path(
            Path('.').parent, 'datasets', 'QuadSDK-A1Speed1.0').absolute()
    train_model_quad_sdk(path_to_urdf, path_to_quad_sdk)

if __name__ == '__main__':
     main()