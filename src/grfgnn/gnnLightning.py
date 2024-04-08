import os
import torch
from torch import optim, nn, utils, Tensor
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
import lightning as L
from lightning.pytorch import seed_everything
from torch_geometric.nn.models import MLP, GCN
from torch_geometric.data.lightning import LightningDataset
from lightning.pytorch.loggers import WandbLogger
from .datasets import CerberusStreetDataset
from .urdfParser import RobotURDF
from torch_geometric.loader import DataLoader


# define the LightningModule
class GCN_Lightning(L.LightningModule):

    def __init__(self, num_node_features, y_indices, num_nodes):
        super().__init__()
        # define any number of nn.Modules (or use your current ones)
        self.gcn_model = GCN(in_channels=num_node_features,
                             hidden_channels=256,
                             num_layers=4,
                             out_channels=1)
        self.y_indices = y_indices
        self.num_nodes = num_nodes
        self.save_hyperparameters()

    def step_helper_function(self, batch, batch_idx):
        out_raw = self.gcn_model(x=batch.x,
                                 edge_index=batch.edge_index).squeeze()

        # Reshape so that we have a tensor of (batch_size, num_nodes)
        out_nodes_by_batch = torch.reshape(out_raw,
                                           (batch.batch_size, self.num_nodes))

        # Get the outputs from the foot nodes
        truth_tensors = []
        for index in self.y_indices:
            truth_tensors.append(out_nodes_by_batch[:, index])
        out_predicted = torch.stack(truth_tensors).swapaxes(0, 1)

        # Get the labels
        batch_y = torch.reshape(batch.y,
                                (batch.batch_size, len(self.y_indices)))

        # Calculate loss
        loss = nn.functional.mse_loss(out_predicted, batch_y)
        return loss

    def training_step(self, batch, batch_idx):
        loss = self.step_helper_function(batch, batch_idx)
        self.log("train_loss", loss, batch_size=batch.batch_size)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self.step_helper_function(batch, batch_idx)
        self.log("val_loss", loss, batch_size=batch.batch_size)

    def test_step(self, batch, batch_idx):
        loss = self.step_helper_function(batch, batch_idx)
        self.log("test_loss", loss, batch_size=batch.batch_size)

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=0.003)
        return optimizer


def train_GNN_model():
    # Load the A1 urdf
    A1_URDF = RobotURDF('urdf_files/A1/a1.urdf', 'package://a1_description/',
                        'unitree_ros/robots/a1_description', True)

    # Load and shuffle the dataset
    dataset = CerberusStreetDataset(
        '/home/dlittleman/state-estimation-gnn/datasets/cerberus_street',
        A1_URDF, None)

    # Calculate the indices in the output where we look for our calculated forces
    ground_truth_indices = []
    name_to_index = A1_URDF.get_node_name_to_index_dict()
    for urdf_name in dataset.ground_truth_urdf_names:
        ground_truth_indices.append(name_to_index[urdf_name])

    # Initialize the Lightning module
    lightning_model = GCN_Lightning(dataset[0].x.shape[1],
                                    ground_truth_indices,
                                    A1_URDF.get_num_nodes())

    # Set model parameters
    batch_size = 100

    # Split the data into training, validation, and testing sets
    rand_seed = 10341885
    rand_gen = torch.Generator().manual_seed(rand_seed)
    train_set, val_set, test_set = torch.utils.data.random_split(
        dataset, [0.7, 0.2, 0.1], generator=rand_gen)
    
    # TODO: VAL AND TEST SHOULD BE DIFFERENT DATASET

    # Create the dataloader and iterate through the batches
    # dataModule = LightningDataset(train_set,
    #                               val_set,
    #                               test_set,
    #                               batch_size=batch_size,
    #                               num_workers=15)
    trainLoader: DataLoader = DataLoader(train_set,
                                         batch_size=100,
                                         shuffle=True,
                                         num_workers=15)
    valLoader: DataLoader = DataLoader(val_set,
                                       batch_size=100,
                                       shuffle=False,
                                       num_workers=15)
    testLoader: DataLoader = DataLoader(test_set,
                                        batch_size=100,
                                        shuffle=False,
                                        num_workers=15)

    # Create Logger
    wandb_logger = WandbLogger(project="grfgnn")
    wandb_logger.experiment.config["batch_size"] = batch_size

    # train the model (hint: here are some helpful Trainer arguments for rapid idea iteration)
    seed_everything(rand_seed, workers=True)
    trainer = L.Trainer(
        default_root_dir="./models/GCN/",
        deterministic=True,  # Reproducability
        devices='auto',
        accelerator="auto",
        #limit_train_batches=100,
        max_epochs=100,
        check_val_every_n_epoch=1,
        enable_progress_bar=True,
        logger=wandb_logger)
    trainer.fit(lightning_model, trainLoader, valLoader)
    trainer.test(lightning_model, dataloaders=testLoader)
