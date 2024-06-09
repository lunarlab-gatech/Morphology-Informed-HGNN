import torch
import torch.nn.functional as F
from torch import optim, nn
import lightning as L
from torch_geometric.nn.models import GAT
from torch_geometric.nn import to_hetero, Linear, HeteroConv, GATv2Conv, HeteroDictLinear
from lightning.pytorch.loggers import WandbLogger
from .datasets_deprecated import CerberusDataset
from .datasets import FlexibleDataset
from torch_geometric.loader import DataLoader
from lightning.pytorch.callbacks import ModelCheckpoint
from pathlib import Path
import names
import matplotlib.pyplot as plt
from torch.utils.data import Subset


def get_foot_node_outputs_gnn(out_raw, batch, y_indices, model_type):
    """
    Helper method that reshapes the raw output of the
    gnn-based models and extracts the foot node so that 
    we can properly calculate the loss.
    """

    second_dim_size = None
    if model_type == "gnn":
        second_dim_size = int(batch.x.shape[0] / batch.batch_size)
    elif model_type == "heterogeneous_gnn":
        second_dim_size = int(batch["foot"].x.shape[0] / batch.batch_size)
    else:
        raise ValueError("Invalid model_type")

    # Reshape so that we have a tensor of (batch_size, num_nodes)
    out_nodes_by_batch = torch.reshape(out_raw.squeeze(), (batch.batch_size, second_dim_size))

    # Get the outputs from the foot nodes
    truth_tensors = []
    for index in y_indices:
        truth_tensors.append(out_nodes_by_batch[:, index])
    y_pred = torch.stack(truth_tensors).swapaxes(0, 1)
    return y_pred

def get_foot_node_labels_gnn(batch, y_indices):
    """
    Helper method that reshapes the labels of the gnn
    datasets so we can properly calculate the loss.
    """
    return torch.reshape(batch.y, (batch.batch_size, len(y_indices)))

class GRF_HGNN(torch.nn.Module):
    """
    The Ground Reaction Force Heterogeneous Graph Neural Network model.
    """

    def __init__(self, hidden_channels, edge_dim, num_layers, out_channels,
                 data_metadata):
        super().__init__()

        # Create the first layer encoder to convert features into embeddings
        # Hopefully for all node and edge features
        # TODO: Test that it encodes both node and edge features
        self.encoder = HeteroDictLinear(-1, hidden_channels, data_metadata[0])

        # Create dictionary that maps edge connections type to convolutional operator
        conv_dict = {}
        for edge_connection in data_metadata[1]:
            # TODO: Maybe wap this out with an operation that better matches what we want to happen with the edges,
            # instead of just using it for the attention calculation.
            conv_dict[edge_connection] = GATv2Conv(hidden_channels,
                                                   hidden_channels,
                                                   add_self_loops=False,
                                                   edge_dim=edge_dim,
                                                   aggr='sum')

        # Create a convolution for each layer
        self.convs = torch.nn.ModuleList()
        for _ in range(num_layers):
            conv = HeteroConv(conv_dict, aggr='sum')
            self.convs.append(conv)

        # Create the final linear layer (Decoder) -> Just for nodes of type "foot"
        # Meant to calculate the final GRF values
        self.decoder = Linear(hidden_channels, out_channels)

    def forward(self, x_dict, edge_index_dict):
        x_dict = self.encoder(x_dict)
        x_dict = {key: x.relu() for key, x in x_dict.items()}
        for conv in self.convs:
            # TODO: Does the RELU actually work?
            x_dict = conv(x_dict, edge_index_dict)
            x_dict = {key: x.relu() for key, x in x_dict.items()}
        return self.decoder(x_dict['foot'])


class Base_Lightning(L.LightningModule):
    """
    Define training, validation, test, and prediction 
    steps used by all models, in addition to the 
    optimizer.
    """

    def log_losses(self, batch_size, mse_loss, y, y_pred, step_name: str):
        self.log(step_name + "_MSE_loss", mse_loss, batch_size=batch_size)
        self.log(step_name + "_RMSE_loss",
                 torch.sqrt(mse_loss),
                 batch_size=batch_size)
        self.log(step_name + "_L1_loss",
                 nn.functional.l1_loss(y, y_pred),
                 batch_size=batch_size)

        # Log losses per individual leg
        for i in range(0, 4):
            y_leg = y[:, i]
            y_pred_leg = y_pred[:, i]
            leg_mse_loss = nn.functional.mse_loss(y_leg, y_pred_leg)
            self.log(step_name + "_MSE_loss:leg_" + str(i),
                     leg_mse_loss,
                     batch_size=batch_size)
            self.log(step_name + "_RMSE_loss:leg_" + str(i),
                     torch.sqrt(leg_mse_loss),
                     batch_size=batch_size)
            self.log(step_name + "_L1_loss:leg_" + str(i),
                     nn.functional.l1_loss(y_leg, y_pred_leg),
                     batch_size=batch_size)

    def training_step(self, batch, batch_idx):
        mse_loss, y, y_pred, batch_size = self.step_helper_function(
            batch, batch_idx)
        self.log_losses(batch_size, mse_loss, y, y_pred, "train")
        return mse_loss

    def validation_step(self, batch, batch_idx):
        mse_loss, y, y_pred, batch_size = self.step_helper_function(
            batch, batch_idx)
        self.log_losses(batch_size, mse_loss, y, y_pred, "val")

    def test_step(self, batch, batch_idx):
        mse_loss, y, y_pred, batch_size = self.step_helper_function(
            batch, batch_idx)
        self.log_losses(batch_size, mse_loss, y, y_pred, "test")

    def predict_step(self, batch, batch_idx):
        mse_loss, y, y_pred, batch_size = self.step_helper_function(
            batch, batch_idx)
        return y_pred, y

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=0.003)
        return optimizer

    def step_helper_function(self, batch, batch_idx):
        """
        Function that actually runs the model on the batch
        to get loss and model output.

        Returns:
            mse_loss: The PyTorch MSE loss between y and y_pred.
            y: The ground truth labels
            y_pred: The predicted model output.
        """
        raise NotImplementedError


class MLP_Lightning(Base_Lightning):

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
            for i in range(0, num_layers - 2):
                modules.append(nn.Linear(hidden_channels, hidden_channels))
                modules.append(nn.ReLU())
            modules.append(nn.Linear(hidden_channels, 4))
            modules.append(nn.ReLU())

        self.mlp_model = nn.Sequential(*modules)
        self.save_hyperparameters()

    def step_helper_function(self, batch, batch_idx):
        x, y = batch
        y_pred = self.mlp_model(x)
        loss = nn.functional.mse_loss(y_pred, y)
        batch_size = x.shape[0]
        return loss, y, y_pred, batch_size


class GNN_Lightning(Base_Lightning):

    def __init__(self, num_node_features, hidden_channels, num_layers,
                 y_indices):
        """
        Constructor for GCN_Lightning class. Pytorch Lightning
        wrapper around the Pytorch geometric GCN class.

        Parameters:
            num_node_features (int) - The number of features each
                node embedding has.
            hidden_channels (int) - The hidden size.
            num_layers (int) - The number of layers in the model.
            y_indices (list[int]) - The indices of the GCN output
                that should match the ground truch labels provided.
                All other node outputs of the GCN are ignored.
        """
        super().__init__()
        self.gnn_model = GAT(in_channels=num_node_features,
                             hidden_channels=hidden_channels,
                             num_layers=num_layers,
                             out_channels=1,
                             v2=True)
        self.y_indices = y_indices
        self.save_hyperparameters()

    def step_helper_function(self, batch, batch_idx):
        # Get the raw output
        out_raw = self.gnn_model(x=batch.x,edge_index=batch.edge_index)

        # Get the outputs from the foot nodes
        y_pred = get_foot_node_outputs_gnn(out_raw, batch, self.y_indices, "gnn")

        # Calculate loss
        y = get_foot_node_labels_gnn(batch, self.y_indices)
        loss = nn.functional.mse_loss(y, y_pred)
        return loss, y, y_pred, batch.batch_size


class Heterogeneous_GNN_Lightning(Base_Lightning):

    def __init__(self, hidden_channels, edge_dim, num_layers, y_indices,
                 data_metadata, dummy_batch):
        """
        Constructor for Heterogeneous GNN.

        Parameters:
            hidden_channels (int) - The hidden size.
            edge_dim (int) - Edge feature dimensionality
            num_layers (int) - The number of layers in the model.
            y_indices (list[int]) - The indices of the output
                that should match the ground truch labels provided.
                All other node outputs of the GNN are ignored.
            data_metadata (tuple) - See https://pytorch-geometric.readthedocs.io/en/latest/modules/nn.html?highlight=to_hetero#torch_geometric.nn.to_hetero_transformer.to_hetero for details.
            dummy_batch - Used to initialize the lazy modules.
        """
        super().__init__()
        self.model = GRF_HGNN(hidden_channels=hidden_channels,
                              edge_dim=edge_dim,
                              num_layers=num_layers,
                              out_channels=1,
                              data_metadata=data_metadata)
        self.y_indices = y_indices

        # Initialize lazy modules
        with torch.no_grad():
            self.model(x_dict=dummy_batch.x_dict,
                       edge_index_dict=dummy_batch.edge_index_dict)
        self.save_hyperparameters()

    def step_helper_function(self, batch, batch_idx):
        # Get the raw foot output
        out_raw = self.model(x_dict=batch.x_dict,
                             edge_index_dict=batch.edge_index_dict)

        # Get the outputs from the foot nodes
        y_pred = get_foot_node_outputs_gnn(out_raw, batch, self.y_indices, "heterogeneous_gnn")

        # Calculate loss
        y = get_foot_node_labels_gnn(batch, self.y_indices)
        loss = nn.functional.mse_loss(y_pred, y)
        return loss, y, y_pred, batch.batch_size


def evaluate_model(path_to_checkpoint: Path, predict_dataset: Subset):
    """
    Runs the provided model on the corresponding dataset,
    and returns the predicted GRF values and the ground truth values.

    Returns:
        pred - Predicted GRF values
        labels - Ground Truth GRF values
    """

    # Set the dtype to be 64 by default
    torch.set_default_dtype(torch.float64)

    # Get the model type
    model_type = predict_dataset.dataset.get_data_format()

    # Initialize the model
    model = None
    if model_type == 'mlp':
        model = MLP_Lightning.load_from_checkpoint(str(path_to_checkpoint))
    elif model_type == 'gnn':
        model = GNN_Lightning.load_from_checkpoint(str(path_to_checkpoint))
    elif model_type == 'heterogeneous_gnn':
        model = Heterogeneous_GNN_Lightning.load_from_checkpoint(str(path_to_checkpoint))
    else:
        raise ValueError("model_type must be gnn, mlp, or heterogeneous_gnn.")

    # Create a validation dataloader
    valLoader: DataLoader = DataLoader(predict_dataset, batch_size=100, shuffle=False, num_workers=15)

    # Predict with the model
    pred = torch.zeros((0, 4))
    labels = torch.zeros((0, 4))
    if model_type != 'heterogeneous_gnn':
        trainer = L.Trainer()
        predictions_result = trainer.predict(model, valLoader)
        for batch_result in predictions_result:
            pred = torch.cat((pred, batch_result[0]), dim=0)
            labels = torch.cat((labels, batch_result[1]), dim=0)

    else:  # for 'heterogeneous_gnn'
        device = 'cpu'  # 'cuda' if torch.cuda.is_available() else
        model.model = model.model.to(device)
        with torch.no_grad():
            for batch in valLoader:
                out_raw = model.model(x_dict=batch.x_dict,
                                      edge_index_dict=batch.edge_index_dict)

                # Get the outputs from the foot nodes
                pred_batch = get_foot_node_outputs_gnn(out_raw, batch, model.y_indices, "heterogeneous_gnn")

                # Get the labels
                labels_batch = get_foot_node_labels_gnn(batch, model.y_indices)

                # Append to the previously collected data
                pred = torch.cat((pred, pred_batch), dim=0)
                labels = torch.cat((labels, labels_batch), dim=0)

    return pred, labels

def train_model(train_dataset: Subset,
                val_dataset: Subset,
                test_dataset: Subset,
                testing_mode: bool = False,
                disable_logger: bool = False,
                batch_size: int = 100,
                num_layers: int = 8):
    """
    Train a learning model with the input datasets. If 
    'testing_mode' is enabled, limit the batches and epoch size
    so that the training completes quickly.

    Returns:
        path_to_save (str) - The path to the checkpoint folder
    """

    # Make sure the underlying datasets have the same data_format
    train_data_format = train_dataset.dataset.get_data_format()
    val_data_format = val_dataset.dataset.get_data_format()
    test_data_format = test_dataset.dataset.get_data_format()
    if train_data_format != val_data_format or val_data_format != test_data_format:
        raise ValueError("Data formats of datasets don't match")

    # Extract important information from the Subsets
    model_type = train_data_format
    ground_truth_label_indices = train_dataset.dataset.get_foot_node_indices_matching_labels(
    )

    # Set appropriate settings for testing mode
    max_epochs = 100
    limit_train_batches = None
    limit_val_batches = None
    limit_test_batches = None
    deterministic = False
    if testing_mode:
        max_epochs = 2
        limit_train_batches = 10
        limit_val_batches = 5
        limit_test_batches = 5
        deterministic = True

    # Set the dtype to be 64 by default
    torch.set_default_dtype(torch.float64)

    # Set model parameters
    hidden_channels = 10

    # Create the dataloaders
    trainLoader: DataLoader = DataLoader(train_dataset,
                                         batch_size=batch_size,
                                         shuffle=True,
                                         num_workers=15)
    valLoader: DataLoader = DataLoader(val_dataset,
                                       batch_size=batch_size,
                                       shuffle=False,
                                       num_workers=15)
    testLoader: DataLoader = DataLoader(test_dataset,
                                        batch_size=batch_size,
                                        shuffle=False,
                                        num_workers=15)

    # Get a dummy_batch
    dummy_batch = None
    for batch in trainLoader:
        dummy_batch = batch
        break

    # Create the model
    lightning_model = None
    if model_type == 'mlp':
        lightning_model = MLP_Lightning(train_dataset[0][0].shape[0],
                                        hidden_channels, num_layers,
                                        batch_size)
    elif model_type == 'gnn':
        lightning_model = GNN_Lightning(train_dataset[0].x.shape[1],
                                        hidden_channels, num_layers,
                                        ground_truth_label_indices)
    elif model_type == 'heterogeneous_gnn':
        lightning_model = Heterogeneous_GNN_Lightning(
            hidden_channels=hidden_channels,
            edge_dim=dummy_batch['base', 'connect',
                                 'joint'].edge_attr.size()[1],
            num_layers=num_layers,
            y_indices=ground_truth_label_indices,
            data_metadata=train_dataset.dataset.get_data_metadata(),
            dummy_batch=dummy_batch)
    else:
        raise ValueError("Invalid model type.")

    # Create Logger
    wandb_logger = False
    run_name = model_type + "-" + names.get_first_name(
        ) + "-" + names.get_last_name()
    if not disable_logger:
        wandb_logger = WandbLogger(project="grfgnn-QuadSDK", name=run_name)
        wandb_logger.watch(lightning_model, log="all")
        wandb_logger.experiment.config["batch_size"] = batch_size

    # Set model parameters
    path_to_save = str(Path("models", run_name))

    # Set up precise checkpointing
    checkpoint_callback = ModelCheckpoint(
        dirpath=path_to_save,
        filename='{epoch}-{val_MSE_loss:.5f}',
        save_top_k=5,
        monitor="val_MSE_loss")

    # Lower precision of operations for faster training
    torch.set_float32_matmul_precision("medium")

    # Train the model and test
    # seed_everything(rand_seed, workers=True)
    trainer = L.Trainer(
        default_root_dir=path_to_save,
        deterministic=deterministic,  # Reproducability
        benchmark=True,
        devices='auto',
        accelerator="auto",
        max_epochs=max_epochs,
        limit_train_batches=limit_train_batches,
        limit_val_batches=limit_val_batches,
        limit_test_batches=limit_test_batches,
        check_val_every_n_epoch=1,
        enable_progress_bar=True,
        logger=wandb_logger,
        callbacks=[checkpoint_callback])
    trainer.fit(lightning_model, trainLoader, valLoader)
    trainer.test(lightning_model, dataloaders=testLoader)

    # Return the path to the trained checkpoint
    return path_to_save
