from pathlib import Path
from mi_hgnn.lightning_py.gnnLightning import train_model, evaluate_model
from mi_hgnn.visualization import visualize_model_outputs_regression
import torch
from mi_hgnn.datasets_py.quadSDKDataset import QuadSDKDataset_A1Speed0_5, QuadSDKDataset_A1Speed1_0, QuadSDKDataset_A1Speed1_5FlippedOver
from torch.utils.data import Subset
import wandb
import yaml


def main():
    # Import the config yaml file
    with open("./research/sweeps/sweep2.yaml") as file:
        config = yaml.load(file, Loader=yaml.FullLoader)
    wandb.init(config=config)

    # TODO: Do we need to use the same A1 URDF file from QuadSDK?
    path_to_urdf = Path('urdf_files', 'A1', 'a1.urdf').absolute()
    path_to_quad_sdk_05 = Path(
        Path('.').parent, 'datasets', 'QuadSDK-A1Speed0.5').absolute()
    path_to_quad_sdk_1 = Path(
        Path('.').parent, 'datasets', 'QuadSDK-A1Speed1.0').absolute()
    path_to_quad_sdk_15Flipped = Path(
        Path('.').parent, 'datasets',
        'QuadSDK-A1Speed1.5FlippedOver').absolute()

    model_type = wandb.config.model_type

    # Initalize the datasets
    dataset_05 = QuadSDKDataset_A1Speed0_5(
        path_to_quad_sdk_05, path_to_urdf, 'package://a1_description/',
        'unitree_ros/robots/a1_description', model_type,
        wandb.config.history_length)
    dataset_1 = QuadSDKDataset_A1Speed1_0(path_to_quad_sdk_1, path_to_urdf,
                                          'package://a1_description/',
                                          'unitree_ros/robots/a1_description',
                                          model_type,
                                          wandb.config.history_length)
    dataset_15Flipped = QuadSDKDataset_A1Speed1_5FlippedOver(
        path_to_quad_sdk_15Flipped, path_to_urdf, 'package://a1_description/',
        'unitree_ros/robots/a1_description', model_type,
        wandb.config.history_length)

    # Split the data into training, validation, and testing sets
    rand_gen = torch.Generator().manual_seed(10341885)
    train_dataset = Subset(dataset_1, range(0, dataset_1.len()))
    val_dataset, test_dataset = torch.utils.data.random_split(
        dataset_05, [0.7, 0.3], generator=rand_gen)

    # Train the model, evaluate, and visualize
    path_to_ckpt = train_model(train_dataset,
                               val_dataset,
                               test_dataset,
                               batch_size=wandb.config.batch_size,
                               num_layers=wandb.config.num_layers,
                               optimizer=wandb.config.optimizer,
                               lr=wandb.config.learning_rate,
                               epochs=wandb.config.epochs,
                               hidden_size=wandb.config.hidden_size)
    models = sorted(Path('.', path_to_ckpt).glob(("epoch=*")))
    pred, labels = evaluate_model(models[0], test_dataset)
    visualize_model_outputs_regression(pred[0:300], labels[0:300], str(path_to_ckpt) + ".pdf")

if __name__ == '__main__':
    main()
