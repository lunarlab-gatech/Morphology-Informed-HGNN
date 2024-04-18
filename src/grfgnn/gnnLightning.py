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
import matplotlib.pyplot as plt


class GCN_Lightning(L.LightningModule):

    def __init__(self, num_node_features, hidden_channels, num_layers, y_indices, num_nodes):
        """
        Constructor for GCN_Lightning class. Pytorch Lightning
        wrapper around the Pytorch geometric GCN class.

        Parameters:
            hidden_channels (int) - The hidden size.
            num_layers (int) - The number of layers in the model.
            num_node_features (int) - The number of features each
                node embedding has.
            y_indices (list[int]) - The indices of the GCN output
                that should match the ground truch labels provided.
                All other node outputs of the GCN are ignored.
            num_nodes (int) - The number of nodes in the graph.
        """
        super().__init__()
        self.gcn_model = GCN(in_channels=num_node_features,
                             hidden_channels=hidden_channels,
                             num_layers=num_layers,
                             out_channels=1)
        self.y_indices = y_indices
        self.num_nodes = num_nodes
        self.save_hyperparameters()

    def get_network_output(self, batch, batch_idx):
        """
        Helper method that takes a batch as input,
        and returns the predicted labels
        """
        out_raw = self.gcn_model(x=batch.x,
                                 edge_index=batch.edge_index).squeeze()

        # Reshape so that we have a tensor of (batch_size, num_nodes)
        out_nodes_by_batch = torch.reshape(out_raw,
                                           (batch.batch_size, self.num_nodes))

        # Get the outputs from the foot nodes
        truth_tensors = []
        for index in self.y_indices:
            truth_tensors.append(out_nodes_by_batch[:, index])
        return torch.stack(truth_tensors).swapaxes(0, 1)

    def predict_step(self, batch, batch_idx):
        y_pred = self.get_network_output(batch, batch_idx)
        y = torch.reshape(batch.y,
                                (batch.batch_size, len(self.y_indices)))
        print("y: ", y)
        print("y_pred: ", y_pred)
        loss = nn.functional.mse_loss(y_pred, y)
        print("mse loss: ", loss)
        l1_loss = nn.functional.l1_loss(y_pred, y)
        print("l1_loss:", l1_loss)
        return y_pred, y

    def step_helper_function(self, batch, batch_idx):
        out_predicted = self.get_network_output(batch, batch_idx)
        batch_y = torch.reshape(batch.y,
                                (batch.batch_size, len(self.y_indices)))
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

    def __init__(self, in_channels, hidden_channels, num_layers, batch_size):
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

        # Create the proper number of layers
        modules = []
        if num_layers < 1:
            raise ValueError("num_layers must be 1 or greater")
        elif num_layers is 1:
            modules.append(nn.Linear(in_channels, 4))
            modules.append(nn.ReLU())
        elif num_layers is 2:
            modules.append(nn.Linear(in_channels, hidden_channels))
            modules.append(nn.ReLU())
            modules.append(nn.Linear(hidden_channels, 4))
            modules.append(nn.ReLU())
        else:
            modules.append(nn.Linear(in_channels, hidden_channels))
            modules.append(nn.ReLU())
            for i in range(0, num_layers-2):
                modules.append(nn.Linear(hidden_channels, hidden_channels))
                modules.append(nn.ReLU())
            modules.append(nn.Linear(hidden_channels, 4))
            modules.append(nn.ReLU())

        self.mlp_model = nn.Sequential(*modules)
        self.save_hyperparameters()
    
    def predict_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self.mlp_model(x)
        print("y: ", y)
        print("y_pred: ", y_pred)
        loss = nn.functional.mse_loss(y_pred, y)
        print("mse loss: ", loss)
        l1_loss = nn.functional.l1_loss(y_pred, y)
        print("l1_loss:", l1_loss)
        return y_pred, y

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

def display_on_axes(axes, estimated, ground_truth, title):
    """
    Simple function that displays grounth truth and estimated
    information on a Matplotlib.pyplot Axes.
    """
    axes.plot(ground_truth, label="Ground Truth", linestyle='-.')
    axes.plot(estimated, label="Estimated")
    axes.legend()
    axes.set_title(title)

def visualize_model_outputs(model_type: str, path_to_checkpoint: Path, 
                            path_to_urdf: Path, path_to_cerberus_track: Path,
                            num_to_visualize: int, path_to_file: Path = None):

    # Load the A1 urdf
    A1_URDF = RobotURDF(path_to_urdf, 'package://a1_description/',
                        'unitree_ros/robots/a1_description', True)

    # Load the test dataset
    street_dataset = CerberusTrackDataset(path_to_cerberus_track,
        A1_URDF, model_type)

    # Initialize the model
    model = None
    if model_type is 'gnn':
        model = GCN_Lightning.load_from_checkpoint(str(path_to_checkpoint))
    elif model_type is 'mlp':
        model = MLP_Lightning.load_from_checkpoint(str(path_to_checkpoint))
    else:
        raise ValueError("model_type must be gnn or mlp.")

    # Create a validation dataloader
    valLoader: DataLoader = DataLoader(street_dataset, batch_size=num_to_visualize, shuffle=False,
                                       num_workers=15)

    # Setup four graphs
    fig, axes = plt.subplots(4, figsize=[20, 10])
    fig.suptitle('Foot Estimated Forces vs. Ground Truth')

    # Validate with the model
    trainer = L.Trainer(limit_predict_batches=1)
    predictions_result = trainer.predict(model, valLoader)
    pred = predictions_result[0][0].numpy()
    labels = predictions_result[0][1].numpy()

    # Display the results
    titles = [
        "Front Left Foot Forces", "Front Right Foot Forces",
        "Rear Left Foot Forces", "Rear Right Foot Forces"
    ]
    for i in range(0, 4):
        display_on_axes(axes[i], pred[:, i], labels[:, i],
                        titles[i])
        
    # Show the figure
    print(path_to_file)
    if path_to_file is not None:
        plt.savefig(path_to_file)
    plt.show()

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
    hidden_channels = 256
    num_layers = 8

    # Create the model
    lightning_model = None
    if model_type is 'gnn':
        lightning_model = GCN_Lightning(
            street_dataset[0].x.shape[1],
            hidden_channels,
            num_layers,
            street_dataset.get_ground_truth_label_indices(),
            A1_URDF.get_num_nodes())
    elif model_type is 'mlp':
        lightning_model = MLP_Lightning(24, hidden_channels, num_layers, batch_size)
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
