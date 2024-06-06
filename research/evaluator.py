from grfgnn.datasets import QuadSDKDataset
from pathlib import Path
from grfgnn.gnnLightning import evaluate_model_and_visualize
import torch

def main():
    # Get important paths
    path_to_a1_urdf = Path(Path('.').parent, 'urdf_files', 'A1',
                        'a1.urdf').absolute()
    path_to_go1_urdf = Path(Path('.').parent, 'urdf_files', 'Go1',
                            'go1.urdf').absolute()
    path_to_cerberus_street = Path(
        Path('.').parent, 'datasets', 'cerberus_street').absolute()
    path_to_cerberus_track = Path(Path('.').parent, 'datasets',
                                'cerberus_track').absolute()
    path_to_cerberus_campus = Path(
        Path('.').parent, 'datasets', 'cerberus_campus').absolute()
    path_to_xiong_simulated = Path(
        Path('.').parent, 'datasets', 'xiong_simulated').absolute()
    path_to_quad_sdk = Path(
            Path('.').parent, 'datasets', 'QuadSDK-A1Speed1.0').absolute()
    
    # Set model types
    model_type = 'heterogeneous_gnn'

    # Give path to checkpoint
    path_to_checkpoint = None
    if model_type == 'mlp':
        path_to_checkpoint = Path(
            Path('.').parent, 'models', 'mlp-Glenn-Wallace',
            'epoch=6-val_loss=5764.05.ckpt')
    elif model_type == 'gnn':
        path_to_checkpoint = Path(
            Path('.').parent, 'models', 'gnn-Estella-Kidd',
            'epoch=15-val_loss=6772.86.ckpt')
    elif model_type == 'heterogeneous_gnn':
        path_to_checkpoint = Path(Path('.').parent, 'models', 'heterogeneous_gnn-Frank-Pettus', 'epoch=35-val_MSE_loss=330.98426.ckpt')

    # Initalize the datasets
    a1_sim_dataset = QuadSDKDataset(
        path_to_quad_sdk, path_to_a1_urdf, 'package://a1_description/',
        'unitree_ros/robots/a1_description', model_type)

    # Split the data into training, validation, and testing sets
    rand_seed = 10341885
    rand_gen = torch.Generator().manual_seed(rand_seed)
    train_size = int(0.7 * a1_sim_dataset.len())
    val_size = int(0.2 * a1_sim_dataset.len())
    test_size = a1_sim_dataset.len() - (train_size + val_size)
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
        a1_sim_dataset, [train_size, val_size, test_size], generator=rand_gen)

    evaluate_model_and_visualize(model_type, path_to_checkpoint, test_dataset, [0, 567], 'visualizations/results.png')

if __name__ == "__main__":
    main()