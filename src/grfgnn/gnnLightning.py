import torch
from torch import optim, nn
import lightning as L
from lightning.pytorch import seed_everything
from torch_geometric.nn.models import GCN
from lightning.pytorch.loggers import WandbLogger
from .datasets import CerberusStreetDataset, CerberusTrackDataset
from .urdfParser import RobotURDF
from torch_geometric.loader import DataLoader
from lightning.pytorch.callbacks import ModelCheckpoint
from pathlib import Path
import names


class GCN_Lightning(L.LightningModule):

    def __init__(self, num_node_features, y_indices, num_nodes):
        """
        Constructor for GCN_Lightning class. Pytorch Lightning
        wrapper around the Pytorch geometric GCN class.

        Parameters:
            num_node_features (int) - The number of features each
                node embedding has.
            y_indices (list[int]) - The indices of the GCN output
                that should match the ground truch labels provided.
                All other node outputs of the GCN are ignored.
            num_nodes (int) - The number of nodes in the graph.
        """
        super().__init__()
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


class MLP_Lightning(L.LightningModule):

    def __init__(self, in_channels, hidden_channels, batch_size):
        """
        Constructor for MLP_Lightning class. Pytorch Lightning
        wrapper around the Pytorch Torchvision MLP class.

        Parameters:
            in_channels (int) - Number of input parameters to the model.
            hidden_channels (int) - The hidden size.
            batch_size (int) - The size of the batches from the dataloaders.
        """
        super().__init__()
        self.batch_size = batch_size
        self.mlp_model = nn.Sequential(
            nn.Linear(in_channels, hidden_channels), nn.ReLU(),
            nn.Linear(hidden_channels, hidden_channels), nn.ReLU(),
            nn.Linear(hidden_channels, hidden_channels), nn.ReLU(),
            nn.Linear(hidden_channels, 4), nn.ReLU())
        self.save_hyperparameters()

    def step_helper_function(self, batch, batch_idx):
        x, y = batch
        y_pred = self.mlp_model(x)
        loss = nn.functional.mse_loss(y_pred, y)
        return loss

    def training_step(self, batch, batch_idx):
        loss = self.step_helper_function(batch, batch_idx)
        self.log("train_loss", loss, batch_size=self.batch_size)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self.step_helper_function(batch, batch_idx)
        self.log("val_loss", loss, batch_size=self.batch_size)

    def test_step(self, batch, batch_idx):
        loss = self.step_helper_function(batch, batch_idx)
        self.log("test_loss", loss, batch_size=self.batch_size)

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=0.003)
        return optimizer


def train_model(path_to_urdf: Path, path_to_cerberus_street: Path, 
                path_to_cerberus_track: Path, model_type: str = 'gnn'):

    # Load the A1 urdf
    A1_URDF = RobotURDF(path_to_urdf, 'package://a1_description/',
                        'unitree_ros/robots/a1_description', True)

    # Initalize the datasets
    street_dataset = CerberusStreetDataset(
        path_to_cerberus_street,
        A1_URDF, model_type)
    track_dataset = CerberusTrackDataset(
        path_to_cerberus_track,
        A1_URDF, model_type)

    # Set batch size
    batch_size = 100

    # Initalize the correct model
    lightning_model = None
    if model_type is 'gnn':
        lightning_model = GCN_Lightning(
            street_dataset[0].x.shape[1],
            street_dataset.get_ground_truth_label_indices(),
            A1_URDF.get_num_nodes())
    elif model_type is 'mlp':
        lightning_model = MLP_Lightning(24, 256, batch_size)
    else:
        raise ValueError("Invalid model type.")

    # Create Logger
    run_name = model_type + "-" + names.get_first_name(
    ) + "-" + names.get_last_name()
    wandb_logger = WandbLogger(project="grfgnn", name=run_name)
    wandb_logger.watch(lightning_model, log="all")

    # Set model parameters
    wandb_logger.experiment.config["batch_size"] = batch_size
    rand_seed = 10341885
    path_to_save = str(Path("models", wandb_logger.experiment.name))
    rand_gen = torch.Generator().manual_seed(rand_seed)

    # Split the data into training, validation, and testing sets
    train_set = street_dataset
    val_size = int(0.7 * track_dataset.len())
    test_size = track_dataset.len() - val_size
    val_set, test_set = torch.utils.data.random_split(track_dataset,
                                                      [val_size, test_size],
                                                      generator=rand_gen)

    # Create the dataloaders
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

    # Set up precise checkpointing
    checkpoint_callback = ModelCheckpoint(dirpath=path_to_save,
                                          filename='{epoch}-{val_loss:.2f}',
                                          save_top_k=5,
                                          monitor="val_loss")

    # Lower precision of operations for faster training
    torch.set_float32_matmul_precision("medium")

    # Train the model and test
    # seed_everything(rand_seed, workers=True)
    trainer = L.Trainer(
        default_root_dir=path_to_save,
        # deterministic=True,  # Reproducability
        benchmark=True, 
        devices='auto',
        accelerator="auto",
        max_epochs=100,
        check_val_every_n_epoch=1,
        enable_progress_bar=True,
        logger=wandb_logger,
        callbacks=[checkpoint_callback])
    trainer.fit(lightning_model, trainLoader, valLoader)
    trainer.test(lightning_model, dataloaders=testLoader)
