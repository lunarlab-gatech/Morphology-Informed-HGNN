from grfgnn.datasets import QuadSDKDataset
from pathlib import Path
from grfgnn.gnnLightning import evaluate_model
from grfgnn import QuadSDKDataset_A1Speed0_5, QuadSDKDataset_A1Speed1_0, QuadSDKDataset_A1Speed1_5FlippedOver
from grfgnn.visualization import visualize_model_outputs
import torch
from torch.utils.data import Subset


def main():
    # Set model types
    model_type = 'heterogeneous_gnn'

    # Give path to checkpoint
    # ================================= FILL THESE IN ===================================
    model_directory_name = "comic-sweep-34"  # Ex: 'gnn-Jared-Wright'
    ckpt_file_name = "epoch=67-val_MSE_loss=196.41758.ckpt"  # Ex: 'epoch=0-val_loss=6544.70.ckpt'
    model_type = 'gnn'
    history_length = 5
    
    # ===================================================================================
    path_to_checkpoint = Path(
        Path('.').parent, 'models', model_directory_name, ckpt_file_name)

    # TODO: Do we need to use the same A1 URDF file from QuadSDK?
    path_to_urdf = Path('urdf_files', 'A1', 'a1.urdf').absolute()
    path_to_quad_sdk_05 = Path(
        Path('.').parent, 'datasets', 'QuadSDK-A1Speed0.5').absolute()
    path_to_quad_sdk_1 = Path(
        Path('.').parent, 'datasets', 'QuadSDK-A1Speed1.0').absolute()
    path_to_quad_sdk_15Flipped = Path(
        Path('.').parent, 'datasets',
        'QuadSDK-A1Speed1.5FlippedOver').absolute()

    # Initalize the datasets
    dataset_05 = QuadSDKDataset_A1Speed0_5(
        path_to_quad_sdk_05, path_to_urdf, 'package://a1_description/',
        'unitree_ros/robots/a1_description', model_type, history_length)
    dataset_1 = QuadSDKDataset_A1Speed1_0(path_to_quad_sdk_1, path_to_urdf,
                                          'package://a1_description/',
                                          'unitree_ros/robots/a1_description',
                                          model_type, history_length)
    dataset_15Flipped = QuadSDKDataset_A1Speed1_5FlippedOver(
        path_to_quad_sdk_15Flipped, path_to_urdf, 'package://a1_description/',
        'unitree_ros/robots/a1_description', model_type, history_length)

    # Split the data into training, validation, and testing sets
    rand_gen = torch.Generator().manual_seed(10341885)
    train_dataset = Subset(dataset_1, range(0, dataset_1.len()))
    val_dataset, test_dataset = torch.utils.data.random_split(
        dataset_05, [0.7, 0.3], generator=rand_gen)

    # ================================= FILL THESE IN ===================================
    num_entries_to_visualize = 400  # Ex: 2000
    # ===================================================================================
    pred, labels = evaluate_model(
        path_to_checkpoint, Subset(dataset_05, range(0, dataset_05.len())))
    visualize_model_outputs(pred[0:num_entries_to_visualize],
                            labels[0:num_entries_to_visualize],
                            str(path_to_checkpoint) + ".pdf")

if __name__ == "__main__":
    main()
