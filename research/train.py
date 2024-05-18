from pathlib import Path
from grfgnn.gnnLightning import train_model, train_model_go1_simulated
import torch
from grfgnn.datasets import CerberusStreetDataset, CerberusTrackDataset, Go1SimulatedDataset

def train_model_cerberus(path_to_urdf, path_to_cerberus_street,
                         path_to_cerberus_track, model_type):

    # Initalize the datasets
    street_dataset = CerberusStreetDataset(
        path_to_cerberus_street, path_to_urdf, 'package://a1_description/',
        'unitree_ros/robots/a1_description', model_type)
    track_dataset = CerberusTrackDataset(path_to_cerberus_track, path_to_urdf,
                                         'package://a1_description/',
                                         'unitree_ros/robots/a1_description',
                                         model_type)

    # Split the data into training, validation, and testing sets
    rand_seed = 10341885
    rand_gen = torch.Generator().manual_seed(rand_seed)
    val_size = int(0.7 * track_dataset.len())
    test_size = track_dataset.len() - val_size
    val_dataset, test_dataset = torch.utils.data.random_split(
        track_dataset, [val_size, test_size], generator=rand_gen)

    # Train the model
    train_model(street_dataset, val_dataset, test_dataset, model_type,
                street_dataset.get_ground_truth_label_indices(), None)


def train_model_go1_simulated(path_to_urdf, path_to_go1_simulated):
    model_type = 'heterogeneous_gnn'

    # Initalize the dataset
    go1_sim_dataset = Go1SimulatedDataset(
        path_to_go1_simulated, path_to_urdf, 'package://go1_description/',
        'unitree_ros/robots/go1_description', model_type)

    # Split the data into training, validation, and testing sets
    rand_seed = 10341885
    rand_gen = torch.Generator().manual_seed(rand_seed)
    train_size = int(0.7 * go1_sim_dataset.len())
    val_size = int(0.2 * go1_sim_dataset.len())
    test_size = go1_sim_dataset.len() - (train_size + val_size)
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
        go1_sim_dataset, [train_size, val_size, test_size], generator=rand_gen)

    # Train the model
    if model_type == 'heterogeneous_gnn':
        train_model(train_dataset, val_dataset, test_dataset, model_type,
                    go1_sim_dataset.get_ground_truth_label_indices(),
                    go1_sim_dataset.get_data_metadata())
    elif model_type == 'mlp':
        train_model(train_dataset, val_dataset, test_dataset, model_type,
                    go1_sim_dataset.get_ground_truth_label_indices(),
                    None)

def main():
        path_to_urdf = Path('urdf_files', 'Go1', 'go1.urdf').absolute()
        path_to_xiong_simulated = Path(
                Path('.').parent, 'datasets', 'xiong_simulated').absolute()
        train_model_go1_simulated(path_to_urdf, path_to_xiong_simulated)

if __name__ == '__main__':
     main()