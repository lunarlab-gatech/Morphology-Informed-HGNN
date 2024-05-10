from grfgnn.datasets import visualize_graph, CerberusTrackDataset, CerberusStreetDataset, CerberusCampusDataset, Go1SimulatedDataset
from grfgnn.graphParser import NormalRobotGraph
from pathlib import Path
from grfgnn.gnnLightning import evaluate_model_and_visualize, train_model
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
    
    # Set model types
    model_type = 'heterogeneous_gnn'

    # Load URDF files
    A1_URDF = NormalRobotGraph(path_to_a1_urdf, 'package://a1_description/',
                            'unitree_ros/robots/a1_description', True)
    GO1_URDF = NormalRobotGraph(path_to_go1_urdf, 'package://go1_description/',
                                'unitree_ros/robots/go1_description', True)

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
        path_to_checkpoint = Path(Path('.').parent, 'models', 'heterogeneous_gnn-Ellen-Olmstead', 'epoch=79-val_MSE_loss=21.42154.ckpt')

    # Initalize the datasets
    go1_sim_dataset = Go1SimulatedDataset(path_to_xiong_simulated, path_to_go1_urdf,
                                        'package://go1_description/',
                                        'unitree_ros/robots/go1_description',
                                        model_type)

    # Split the data into training, validation, and testing sets
    rand_seed = 10341885
    rand_gen = torch.Generator().manual_seed(rand_seed)
    train_size = int(0.7 * go1_sim_dataset.len())
    val_size = int(0.2 * go1_sim_dataset.len())
    test_size = go1_sim_dataset.len() - (train_size + val_size)
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
        go1_sim_dataset, [train_size, val_size, test_size], generator=rand_gen)

    evaluate_model_and_visualize(model_type, path_to_checkpoint, test_dataset, [0, 100], 'visualizations/results.png')

if __name__ == "__main__":
    main()