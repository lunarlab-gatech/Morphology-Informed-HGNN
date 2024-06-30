import re
import torch
from torch import optim, nn
import lightning as L
from torch_geometric.nn.models import GAT
from torch_geometric.nn import Linear, HeteroConv, GATv2Conv, HeteroDictLinear
from lightning.pytorch.loggers import WandbLogger
from torch_geometric.loader import DataLoader
from lightning.pytorch.callbacks import ModelCheckpoint
from pathlib import Path
import names
from torch.utils.data import Subset
import torchmetrics
import numpy as np
from .customMetrics import CrossEntropyLossMetric

def get_foot_node_outputs_gnn(out_raw, batch, y_indices, model_type):
    """
    Helper method that reshapes the raw output of the
    gnn-based models and extracts the foot node so that 
    we can properly calculate the loss.
    """

    second_dim_size = None
    shape = out_raw.shape
    if model_type == "gnn":
        second_dim_size = int(batch.x.shape[0] / batch.batch_size)
    elif model_type == "heterogeneous_gnn":
        second_dim_size = int(shape[0] * shape[1] / batch.batch_size)
    else:
        raise ValueError("Invalid model_type")

    # Reshape so that we have a tensor of (batch_size, num_nodes)
    out_nodes_by_batch = torch.reshape(out_raw.squeeze(), (batch.batch_size, second_dim_size))

    # Get the outputs from the foot nodes
    truth_tensors = []
    for index in y_indices:
        start_ind = int(shape[1] * index)
        end_ind = int(start_ind + shape[1])
        truth_tensors.append(out_nodes_by_batch[:, start_ind:end_ind])
    y_pred = torch.cat(truth_tensors, dim=1)
    return y_pred

def get_foot_node_labels_gnn(batch, y_indices):
    """
    Helper method that reshapes the labels of the gnn
    datasets so we can properly calculate the loss.
    """
    return torch.reshape(batch.y, (batch.batch_size, len(y_indices)))


def gnn_classification_output(y: torch.Tensor, y_pred: torch.Tensor):
    """
    Convert the y labels from Bx[4] individual foot contact classes
    into a single Bx1 class out of 16 options.

    Also, convert the y_pred from Bx[4x2] individual foot contact probabilities
    into a single Bx16 class probability which comprises the four states.
    """
    
    # Convert y labels from four sets of 2 classes to one set of 16 classes
    y_np = y.cpu().numpy()
    y_new = np.zeros((y.shape[0], 1))
    for i in range(0, y.shape[0]):
        y_new[i] = y_np[i,0] * 8 + y_np[i,1] * 4 + y_np[i,2] * 2  + y_np[i,3]
    y_new = torch.tensor(y_new, dtype=int)

    # Convert y_pred from two class predictions per foot to a single 16 class prediction
    y_pred_new = torch.zeros((y_pred.shape[0], 16))
    for i in range(0, y_pred.shape[0]):
        for j in range(0, 16):
            foot_0_prob = y_pred[i,int(np.floor(j / 8.0) % 2)]
            foot_1_prob = y_pred[i,int((np.floor(j / 4.0) % 2) + 2)]
            foot_2_prob = y_pred[i,int((np.floor(j / 2.0) % 2) + 4)]
            foot_3_prob = y_pred[i,int((j % 2) + 6)]
            y_pred_new[i,j] = torch.mul(torch.mul(foot_0_prob, foot_1_prob), torch.mul(foot_2_prob, foot_3_prob))

    return y_new, y_pred_new

class GRF_HGNN(torch.nn.Module):
    """
    The Ground Reaction Force Heterogeneous Graph Neural Network model.
    """

    def __init__(self, hidden_channels: int, edge_dim: int, num_layers: int, 
                 out_channels: int, data_metadata, regression: bool = True, 
                 activation_fn = nn.ReLU()):
        """
        Parameters:
            out_channels (int): Only used for regression problems. The number
                of predicted output values.
            out_classes (int): Only used for classification problems. The number
                of predicted output classes PER FOOT. Ex: if out_classes is 2, 
                then the total # of output classes for all four legs is 2^4, or 16.
        """


        super().__init__()
        self.regression = regression

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

        # Save the activation function
        self.activation = activation_fn

        # Create the final linear layer (Decoder) -> Just for nodes of type "foot"
        # Meant to calculate the final GRF values
        if self.regression:
            self.decoder = Linear(hidden_channels, out_channels)
        else: # Add a Softmax for classification problems
            modules = []
            modules.append(nn.Linear(hidden_channels, 2))
            modules.append(nn.LogSoftmax(dim=1))
            self.decoder = nn.Sequential(*modules)

    def forward(self, x_dict, edge_index_dict):
        x_dict = self.encoder(x_dict)
        x_dict = {key: self.activation(x) for key, x in x_dict.items()}
        for conv in self.convs:
            # TODO: Does the Activation function actually work?
            x_dict = conv(x_dict, edge_index_dict)
            x_dict = {key: self.activation(x) for key, x in x_dict.items()}
        return self.decoder(x_dict['foot'])


class Base_Lightning(L.LightningModule):
    """
    Define training, validation, test, and prediction 
    steps used by all models, in addition to the 
    optimizer.
    """

    def __init__(self, optimizer: str, lr: float, regression: bool):
        """
        Parameters:
            optimizer (str) - A string representing the optimizer to use. 
                Currently supports "adam" and "sgd".
            lr (float) - The learning rate of the optimizer.
            regression (bool) - If true, use regression losses. If false,
                use classification losses.
        """

        # Setup input parameters
        super().__init__()
        self.optimizer = optimizer
        self.lr = lr
        self.regression = regression

        # Setup the metrics
        self.metric_mse: torchmetrics.MeanSquaredError = torchmetrics.regression.MeanSquaredError(squared=True)
        self.metric_rmse: torchmetrics.MeanSquaredError = torchmetrics.regression.MeanSquaredError(squared=False)
        self.metric_l1: torchmetrics.MeanAbsoluteError = torchmetrics.regression.MeanAbsoluteError()
        self.metric_ce = CrossEntropyLossMetric()

        # Setup variables to hold the losses
        self.mse_loss = None
        self.rmse_loss = None
        self.l1_loss = None
        self.ce_loss = None

    # ======================= Logging =======================
    def log_losses(self, step_name: str, on_step: bool):
        # Ensure one is enabled and one is disabled
        on_epoch = not on_step

        # Log the losses
        if self.regression:
            self.log(step_name + "_MSE_loss", 
                    self.mse_loss,
                    on_step=on_step,
                    on_epoch=on_epoch)
            self.log(step_name + "_RMSE_loss",
                    self.rmse_loss,
                    on_step=on_step,
                    on_epoch=on_epoch)
            self.log(step_name + "_L1_loss",
                    self.l1_loss,
                    on_step=on_step,
                    on_epoch=on_epoch)
        else:
            self.log(step_name + "_CE_loss", 
                    self.ce_loss,
                    on_step=on_step,
                    on_epoch=on_epoch)
    
    # ======================= Loss Calculation =======================
    def calculate_losses_step(self, y: torch.Tensor, y_pred: torch.Tensor) -> None:
        if self.regression:
            y = y.flatten()
            y_pred = y_pred.flatten()
            self.mse_loss = self.metric_mse(y, y_pred)
            self.rmse_loss = self.metric_rmse(y, y_pred)
            self.l1_loss = self.metric_l1(y, y_pred)
        else:
            self.ce_loss = self.metric_ce(y.squeeze(), y_pred)
    
    def calculate_losses_epoch(self) -> None:
        if self.regression:
            self.mse_loss = self.metric_mse.compute()
            self.rmse_loss = self.metric_rmse.compute()
            self.l1_loss = self.metric_l1.compute()
        else:
            self.ce_loss = self.metric_ce.compute()
    
    def reset_all_metrics(self) -> None:
        self.metric_mse.reset()
        self.metric_rmse.reset()
        self.metric_l1.reset()
        self.metric_ce.reset()

    # ======================= Training =======================
    def training_step(self, batch, batch_idx):
        y, y_pred, batch_size = self.step_helper_function(batch, batch_idx)
        self.calculate_losses_step(y, y_pred)
        self.log_losses("train", on_step=True)
        if self.regression:
            return self.mse_loss
        else:
            return self.ce_loss
    
    # ======================= Validation =======================
    def on_validation_epoch_start(self):
        self.reset_all_metrics()
    
    def validation_step(self, batch, batch_idx):
        y, y_pred, batch_size = self.step_helper_function(
            batch, batch_idx)
        self.calculate_losses_step(y, y_pred)
        if self.regression:
            return self.mse_loss
        else:
            return self.ce_loss
    
    def on_validation_epoch_end(self):
        self.calculate_losses_epoch()
        self.log_losses("val", on_step=False)

    # ======================= Testing =======================
    def on_test_epoch_start(self):
        self.reset_all_metrics()

    def test_step(self, batch, batch_idx):
        y, y_pred, batch_size = self.step_helper_function(
            batch, batch_idx)
        self.calculate_losses_step(y, y_pred)
        if self.regression:
            return self.mse_loss
        else:
            return self.ce_loss
    
    def on_test_epoch_end(self):
        self.calculate_losses_epoch()
        self.log_losses("test", on_step=False)

    # ======================= Prediction =======================
    def predict_step(self, batch, batch_idx):
        y, y_pred, batch_size = self.step_helper_function(
            batch, batch_idx)
        if not self.regression:
            y, y_pred = gnn_classification_output(y, y_pred)
        return y, y_pred

    # ======================= Optimizer =======================
    def configure_optimizers(self):
        if self.optimizer == "adam":
            optimizer = optim.Adam(self.parameters(), lr=self.lr)
        elif self.optimizer == "sgd":
            optimizer = optim.SGD(self.parameters(), lr=self.lr)
        else:
            raise ValueError("Invalid optimizer setting")
        return optimizer

    # ======================= Helper Functions =======================
    def step_helper_function(self, batch, batch_idx):
        """
        Function that actually runs the model on the batch
        to get loss and model output.

        Returns:
            y: The ground truth labels
            y_pred: The predicted model output.
            batch_size: The batch size.
        """
        raise NotImplementedError


class MLP_Lightning(Base_Lightning):

    def __init__(self, in_channels: int, hidden_channels: int, 
                 out_channels: int, num_layers: int, 
                 batch_size: int, optimizer: str = "adam", lr: float = 0.003,
                 regression: bool = True, activation_fn = nn.ReLU()):
        """
        Constructor for MLP_Lightning class. Pytorch Lightning
        wrapper around the Pytorch Torchvision MLP class.

        Parameters:
            in_channels (int) - Number of input parameters to the model.
            hidden_channels (int) - The hidden size.
            out_channels (int) - The number of outputs from the MLP.
            num_layers (int) - The number of layers in the model.
            batch_size (int) - The size of the batches from the dataloaders.
        """

        super().__init__(optimizer, lr, regression)
        self.batch_size = batch_size
        self.regression = regression
        if regression is False:
            raise ValueError("MLP_Lightning currently only supports regressions problems.")

        # Create the proper number of layers
        modules = []
        if num_layers < 2:
            raise ValueError("num_layers must be 2 or greater")
        elif num_layers is 2:
            modules.append(nn.Linear(in_channels, hidden_channels))
            modules.append(activation_fn)
            modules.append(nn.Linear(hidden_channels, out_channels))
        else:
            modules.append(nn.Linear(in_channels, hidden_channels))
            modules.append(activation_fn)
            for i in range(0, num_layers - 2):
                modules.append(nn.Linear(hidden_channels, hidden_channels))
                modules.append(activation_fn)
            modules.append(nn.Linear(hidden_channels, out_channels))

        self.mlp_model = nn.Sequential(*modules)
        self.save_hyperparameters()

    def step_helper_function(self, batch, batch_idx):
        x, y = batch
        y_pred = self.mlp_model(x)
        batch_size = x.shape[0]
        return y, y_pred, batch_size

class GNN_Lightning(Base_Lightning):

    def __init__(self, num_node_features: int, hidden_channels: int, 
                 num_layers: int, y_indices: list[int], 
                 optimizer: str = "adam", lr: float = 0.003,
                 regression: bool = True, activation_fn = nn.ReLU()):
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
        super().__init__(optimizer, lr, regression)
        self.gnn_model = GAT(in_channels=num_node_features,
                             hidden_channels=hidden_channels,
                             num_layers=num_layers,
                             out_channels=1,
                             v2=True,
                             act=activation_fn)
        self.y_indices = y_indices
        if regression is False:
            raise ValueError("GNN_Lightning currently only supports regressions problems.")
        self.save_hyperparameters()

    def step_helper_function(self, batch, batch_idx):
        # Get the raw output
        out_raw = self.gnn_model(x=batch.x,edge_index=batch.edge_index)

        # Get the outputs from the foot nodes
        y_pred = get_foot_node_outputs_gnn(out_raw, batch, self.y_indices, "gnn")

        # Get the labels
        y = get_foot_node_labels_gnn(batch, self.y_indices)

        return y, y_pred, batch.batch_size


class Heterogeneous_GNN_Lightning(Base_Lightning):

    def __init__(self, hidden_channels: int, edge_dim: int, 
                 num_layers: int, y_indices: list[int],
                 data_metadata, dummy_batch, 
                 optimizer: str = "adam", lr: float = 0.003,
                 regression: bool = True, activation_fn = nn.ReLU()):
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
        super().__init__(optimizer, lr, regression)
        self.model = GRF_HGNN(hidden_channels=hidden_channels,
                              edge_dim=edge_dim,
                              num_layers=num_layers,
                              out_channels=1,
                              data_metadata=data_metadata,
                              regression=regression,
                              activation_fn=activation_fn)
        self.y_indices = y_indices
        self.regression = regression

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

        # Get the labels
        y = get_foot_node_labels_gnn(batch, self.y_indices)

        # If a classification problem, convert from Bx[4x2] to Bx4x2, and then to a Bx16 class probabilities
        if not self.regression:
            y, y_pred = gnn_classification_output(y, y_pred)
        return y, y_pred, batch.batch_size


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
    pred = None
    labels = None
    if model.regression:
        pred = torch.zeros((0, 4))
        labels = torch.zeros((0, 4))
    else:
        pred = torch.zeros((0))
        labels = torch.zeros((0))
    if model_type != 'heterogeneous_gnn':
        trainer = L.Trainer()
        predictions_result = trainer.predict(model, valLoader)
        for batch_result in predictions_result:
            labels = torch.cat((labels, batch_result[0]), dim=0)
            pred = torch.cat((pred, batch_result[1]), dim=0)

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

                if not model.regression:
                    labels_batch, pred_batch = gnn_classification_output(labels_batch, pred_batch)
                    pred_batch = torch.argmax(pred_batch, dim=1)

                # Append to the previously collected data
                pred = torch.cat((pred, pred_batch), dim=0)
                labels = torch.cat((labels, labels_batch), dim=0)

    return pred, labels

def train_model(train_dataset: Subset,
                val_dataset: Subset,
                test_dataset: Subset,
                testing_mode: bool = False,
                disable_logger: bool = False,
                logger_project_name: str = None,
                batch_size: int = 100,
                num_layers: int = 8,
                optimizer: str = "adam", 
                lr: float = 0.003,
                epochs: int = 100,
                hidden_size: int = 10,
                regression: bool = True):
    """
    Train a learning model with the input datasets. If 
    'testing_mode' is enabled, limit the batches and epoch size
    so that the training completes quickly.

    Returns:
        path_to_save (str) - The path to the checkpoint folder
    """

    # Make sure the underlying datasets have the same data_format
    # Assume that they are all Subsets, and that the underlying dataset
    # is all the same type.
    train_data_format, val_data_format, test_data_format = None, None, None
    if isinstance(train_dataset.dataset, torch.utils.data.ConcatDataset):
        train_data_format = train_dataset.dataset.datasets[0].dataset.get_data_format()
        val_data_format = val_dataset.dataset.datasets[0].dataset.get_data_format()
        test_data_format = test_dataset.dataset.datasets[0].get_data_format()
    elif isinstance(train_dataset.dataset, torch.utils.data.Dataset):
        train_data_format = train_dataset.dataset.get_data_format()
        val_data_format = val_dataset.dataset.get_data_format()
        test_data_format = test_dataset.dataset.get_data_format()
    else:
        raise ValueError("Unexpected Data format")
    if train_data_format != val_data_format or val_data_format != test_data_format:
        raise ValueError("Data formats of datasets don't match")

    # Extract important information from the Subsets
    model_type = train_data_format
    ground_truth_label_indices = None
    if isinstance(train_dataset.dataset, torch.utils.data.ConcatDataset):
        ground_truth_label_indices = train_dataset.dataset.datasets[0].dataset.get_foot_node_indices_matching_labels()
    elif isinstance(train_dataset.dataset, torch.utils.data.Dataset):
        ground_truth_label_indices = train_dataset.dataset.get_foot_node_indices_matching_labels()

    data_metadata = None
    if model_type == 'heterogeneous_gnn':
        if isinstance(train_dataset.dataset, torch.utils.data.ConcatDataset):
            data_metadata = train_dataset.dataset.datasets[0].dataset.get_data_metadata()
        elif isinstance(train_dataset.dataset, torch.utils.data.Dataset):
            data_metadata = train_dataset.dataset.get_data_metadata(),

    # Set appropriate settings for testing mode
    limit_train_batches = None
    limit_val_batches = None
    limit_test_batches = None
    deterministic = False
    if testing_mode:
        epochs = 2
        limit_train_batches = 10
        limit_val_batches = 5
        limit_test_batches = 5
        deterministic = True

    # Set the dtype to be 64 by default
    torch.set_default_dtype(torch.float64)

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
                                        hidden_size, 4, num_layers,
                                        batch_size, optimizer, lr, 
                                        regression)
    elif model_type == 'gnn':
        lightning_model = GNN_Lightning(train_dataset[0].x.shape[1],
                                        hidden_size, num_layers,
                                        ground_truth_label_indices, 
                                        optimizer, lr, regression)
    elif model_type == 'heterogeneous_gnn':
        lightning_model = Heterogeneous_GNN_Lightning(
            hidden_channels=hidden_size,
            edge_dim=dummy_batch['base', 'connect',
                                 'joint'].edge_attr.size()[1],
            num_layers=num_layers,
            y_indices=ground_truth_label_indices,
            data_metadata=data_metadata,
            dummy_batch=dummy_batch,
            optimizer=optimizer,
            lr=lr, 
            regression=regression)
    else:
        raise ValueError("Invalid model type.")

    # Create Logger
    wandb_logger = False
    path_to_save = None
    if not disable_logger:
        if logger_project_name is None:
            raise ValueError("Need to define \"logger_project_name\" if logger is enabled.")
        wandb_logger = WandbLogger(project=logger_project_name)
        wandb_logger.watch(lightning_model, log="all")
        wandb_logger.experiment.config["batch_size"] = batch_size
        path_to_save = str(Path("models", wandb_logger.experiment.name))
    else:
        path_to_save = str(Path("models", model_type + "_" + names.get_full_name()))

    # Set up precise checkpointing
    monitor = None
    if regression:
        monitor = "val_MSE_loss"
    else:
        monitor = "val_CE_loss"
    checkpoint_callback = ModelCheckpoint(
        dirpath=path_to_save,
        filename='{epoch}-{val_MSE_loss:.5f}',
        save_top_k=5,
        monitor=monitor)

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
        max_epochs=epochs,
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
