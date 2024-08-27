from lib2to3.pytree import Base
from lightning.pytorch.callbacks import DeviceStatsMonitor
import torch
from torch import optim, nn
import lightning as L
from lightning.pytorch.loggers import WandbLogger
from torch_geometric.loader import DataLoader
from lightning.pytorch.callbacks import ModelCheckpoint
from pathlib import Path
import names
from torch.utils.data import Subset
import numpy as np
import torchmetrics
import torchmetrics.classification
from .customMetrics import CrossEntropyLossMetric
import pinocchio as pin
from .hgnn import GRF_HGNN

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
        self.metric_acc = torchmetrics.Accuracy(task="multiclass", num_classes=16)
        self.metric_f1_leg0 = torchmetrics.classification.BinaryF1Score()
        self.metric_f1_leg1 = torchmetrics.classification.BinaryF1Score()
        self.metric_f1_leg2 = torchmetrics.classification.BinaryF1Score()
        self.metric_f1_leg3 = torchmetrics.classification.BinaryF1Score()

        # Setup variables to hold the losses
        self.mse_loss = None
        self.rmse_loss = None
        self.l1_loss = None
        self.ce_loss = None
        self.acc = None
        self.f1_leg0 = None
        self.f1_leg1 = None
        self.f1_leg2 = None
        self.f1_leg3 = None

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
            self.log(step_name + "_Accuracy", 
                    self.acc,
                    on_step=on_step,
                    on_epoch=on_epoch)
            self.log(step_name + "_F1_Score_Leg_Avg", 
                    (self.f1_leg0 + self.f1_leg1 + self.f1_leg2 + self.f1_leg3) / 4.0,
                    on_step=on_step,
                    on_epoch=on_epoch)
            self.log(step_name + "_F1_Score_Leg_0", 
                    self.f1_leg0,
                    on_step=on_step,
                    on_epoch=on_epoch)
            self.log(step_name + "_F1_Score_Leg_1", 
                    self.f1_leg1,
                    on_step=on_step,
                    on_epoch=on_epoch)
            self.log(step_name + "_F1_Score_Leg_2", 
                    self.f1_leg2,
                    on_step=on_step,
                    on_epoch=on_epoch)
            self.log(step_name + "_F1_Score_Leg_3", 
                    self.f1_leg3,
                    on_step=on_step,
                    on_epoch=on_epoch)
    
    # ======================= Loss Calculation =======================
    def calculate_losses_step(self, y: torch.Tensor, y_pred: torch.Tensor) -> None:
        if self.regression:
            y = y.flatten()
            y_pred = y_pred.flatten()
            self.mse_loss = self.metric_mse(y_pred, y)
            self.rmse_loss = self.metric_rmse(y_pred, y)
            self.l1_loss = self.metric_l1(y_pred, y)
        else:
            batch_size = y_pred.shape[0]

            # Make 2 binary class floats for CE_loss
            y_long = y.long()
            #print("y: ", y[0:5])
            #print("y_pred: ", y_pred[0:5])
            y_pred_2_floats = torch.cat((torch.sub(1, torch.flatten(y_pred)).unsqueeze(dim=1), 
                                           torch.flatten(y_pred).unsqueeze(dim=1)), dim=1)
            #print("y_pred_2_floats: ", y_pred_2_floats[0:20])
            #print("y_long: ", y_long.flatten()[0:20])
            
            self.ce_loss = self.metric_ce(y_pred_2_floats, y_long.flatten())

            # Calculate 16 class predictions for accuracy
            y_pred_16, y_16 = self.classification_conversion_16_class(y_pred, y)
            #print("y_pred_16: ", y_pred_16[0:5])
            #print("y_pred_16 max: ", torch.argmax(y_pred_16, dim=1)[0:5])
            #print("y_16: ", y_16.squeeze()[0:5])
            self.acc = self.metric_acc(torch.argmax(y_pred_16, dim=1), y_16.squeeze())

            # Calculate binary class predictions for f1-scores
            y_pred_2 = torch.reshape(torch.argmax(y_pred_2_floats, dim=1), (batch_size, 4))
            #print("y_pred_2: ", y_pred_2[0:5])
            #print("y: ", y[0:5])
            self.f1_leg0 = self.metric_f1_leg0(y_pred_2[:,0], y[:,0])
            self.f1_leg1 = self.metric_f1_leg1(y_pred_2[:,1], y[:,1])
            self.f1_leg2 = self.metric_f1_leg2(y_pred_2[:,2], y[:,2])
            self.f1_leg3 = self.metric_f1_leg3(y_pred_2[:,3], y[:,3])
    
    def calculate_losses_epoch(self) -> None:
        if self.regression:
            self.mse_loss = self.metric_mse.compute()
            self.rmse_loss = self.metric_rmse.compute()
            self.l1_loss = self.metric_l1.compute()
        else:
            self.ce_loss = self.metric_ce.compute()
            self.acc = self.metric_acc.compute()
            self.f1_leg0 = self.metric_f1_leg0.compute()
            self.f1_leg1 = self.metric_f1_leg1.compute()
            self.f1_leg2 = self.metric_f1_leg2.compute()
            self.f1_leg3 = self.metric_f1_leg3.compute()
    
    def reset_all_metrics(self) -> None:
        self.metric_mse.reset()
        self.metric_rmse.reset()
        self.metric_l1.reset()
        self.metric_ce.reset()
        self.metric_acc.reset()
        self.metric_f1_leg0.reset()
        self.metric_f1_leg1.reset()
        self.metric_f1_leg2.reset()
        self.metric_f1_leg3.reset()

    # ======================= Training =======================
    def training_step(self, batch, batch_idx):
        y, y_pred = self.step_helper_function(batch)
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
        y, y_pred = self.step_helper_function(batch)
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
        y, y_pred = self.step_helper_function(batch)
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
        y, y_pred = self.step_helper_function(batch)
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
    def step_helper_function(self, batch):
        """
        Function that actually runs the model on the batch
        to get loss and model output.

        Returns:
            y: The ground truth labels with the shape (batch_size, 4)
            y_pred: The predicted model output with the shape (batch_size, 4).
                    If self.regression, these are just GRF values. If not 
                    self.regression, then these are the probabilities of contact
                    for each foot (where 1 is 100% probability of contact).

        """
        raise NotImplementedError
    
    def classification_conversion_16_class(self, y_pred: torch.Tensor, y: torch.Tensor):
        """
        Convert the y labels from individual foot contact classes into a single 
        class out of 16 options. In other words, convert y from (batch_size, 4) to
        (batch_size, 1).

        Also, convert the y_pred from individual foot contact probabilities
        into a 16 class probability which comprises the four states. In other words,
        convert y_pred from (batch_size, 4) to (batch_size, 16).

        Parameters:
            y_pred (torch.Tensor) - A Tensor of shape (batch_size, 4)
            y      (torch.Tensor) - A Tensor of shape (batch_size, 4)
        """
        
        # Convert y labels from four sets of 2 classes to one set of 16 classes
        y_np = y.cpu().numpy()
        y_new = np.zeros((y.shape[0], 1))
        for i in range(0, y.shape[0]):
            y_new[i] = y_np[i,0] * 8 + y_np[i,1] * 4 + y_np[i,2] * 2 + y_np[i,3]
        y_new = torch.tensor(y_new, dtype=int)

        # Convert y_pred from two class predictions per foot to a single 16 class prediction
        y_pred_new = torch.zeros((y_pred.shape[0], 16))
        for j in range(0, 16):
            if (np.floor(j / 8.0) % 2): foot_0_prob = y_pred[:,0]
            else: foot_0_prob = 1 - y_pred[:,0]

            if (np.floor(j / 4.0) % 2): foot_1_prob = y_pred[:,1]
            else: foot_1_prob = 1 - y_pred[:,1]

            if (np.floor(j / 2.0) % 2): foot_2_prob = y_pred[:,2]
            else: foot_2_prob = 1 - y_pred[:,2]

            if j % 2: foot_3_prob = y_pred[:,3]
            else: foot_3_prob = 1 - y_pred[:,3]

            y_pred_new[:,j] = torch.mul(torch.mul(foot_0_prob, foot_1_prob), torch.mul(foot_2_prob, foot_3_prob))

        return y_pred_new, y_new


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

        # Create the proper number of layers
        modules = []
        if num_layers < 2:
            raise ValueError("num_layers must be 2 or greater")
        elif num_layers is 2:
            modules.append(nn.Linear(in_channels, hidden_channels))
            modules.append(activation_fn)
            modules.append(nn.Linear(hidden_channels, out_channels))
            if not self.regression:
                modules.append(nn.Sigmoid())
        else:
            modules.append(nn.Linear(in_channels, hidden_channels))
            modules.append(activation_fn)
            for i in range(0, num_layers - 2):
                modules.append(nn.Linear(hidden_channels, hidden_channels))
                modules.append(activation_fn)
            modules.append(nn.Linear(hidden_channels, out_channels))
            if not self.regression:
                modules.append(nn.Sigmoid())

        self.mlp_model = nn.Sequential(*modules)
        self.save_hyperparameters()

    def step_helper_function(self, batch):
        x, y = batch
        y_pred = self.mlp_model(x)
        return y, y_pred

class Heterogeneous_GNN_Lightning(Base_Lightning):

    def __init__(self, hidden_channels: int, edge_dim: int, 
                 num_layers: int, data_metadata, dummy_batch, 
                 optimizer: str = "adam", lr: float = 0.003,
                 regression: bool = True, activation_fn = nn.ReLU()):
        """
        Constructor for Heterogeneous GNN.

        Parameters:
            hidden_channels (int) - The hidden size.
            edge_dim (int) - Edge feature dimensionality
            num_layers (int) - The number of layers in the model.
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
        self.regression = regression

        # Initialize lazy modules
        with torch.no_grad():
            self.model(x_dict=dummy_batch.x_dict,
                       edge_index_dict=dummy_batch.edge_index_dict)
        self.save_hyperparameters()

    def step_helper_function(self, batch):
        # Get the raw foot output
        out_raw = self.model(x_dict=batch.x_dict,
                             edge_index_dict=batch.edge_index_dict)

        # Get the outputs from the foot nodes
        y_pred = torch.reshape(out_raw.squeeze(), (batch.batch_size, 4))

        # Get the labels
        y = torch.reshape(batch.y, (batch.batch_size, 4))
        return y, y_pred
    
class Full_Dyamics_Model_Lightning(Base_Lightning):

    def __init__(self, urdf_model_path: Path, urdf_dir: Path):
        """
        Constructor for a Full Dynamics Model using the Equations
        of motion calculated using Lagrangian Mechanics.

        There is not any learned parameters here. This wrapper simply
        lets us reuse the same metric calculations.

        Only supports regression problems.

        Parameters:
            urdf_model_path (Path) - The path to the urdf file that the model
                will be constructed from.
            urdf_dir (Path) - The path to the urdf file directory
        """

        # Note we set optimizer and lr, but they are unused here
        # as we don't instantiate any underlying model.
        super().__init__("adam", 0.001, regression=True)
        self.regression = True
        self.save_hyperparameters()

        # Build the pinnochio model
        self.model, collision_model, visual_model = pin.buildModelsFromUrdf(
            str(urdf_model_path), str(urdf_dir), pin.JointModelFreeFlyer(), verbose= True
        )
        self.data = self.model.createData()

    def step_helper_function(self, batch):
        # Get the raw foot output
        out_raw = self.model(x_dict=batch.x_dict,
                             edge_index_dict=batch.edge_index_dict)

        # Get the outputs from the foot nodes
        y_pred = torch.reshape(out_raw.squeeze(), (batch.batch_size, 4))

        # Get the labels
        y = torch.reshape(batch.y, (batch.batch_size, 4))
        return y, y_pred


def evaluate_model(path_to_checkpoint: Path, predict_dataset: Subset, num_to_vis: int = 1000):
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
    model_type = None
    if isinstance(predict_dataset.dataset, torch.utils.data.ConcatDataset):
        model_type = predict_dataset.dataset.datasets[0].get_data_format()
    elif isinstance(predict_dataset.dataset, torch.utils.data.Dataset):
        model_type = predict_dataset.dataset.get_data_format()

    # Initialize the model
    model = None
    if model_type == 'mlp':
        model = MLP_Lightning.load_from_checkpoint(str(path_to_checkpoint))
    elif model_type == 'heterogeneous_gnn':
        model = Heterogeneous_GNN_Lightning.load_from_checkpoint(str(path_to_checkpoint))
    else:
        raise ValueError("model_type must be mlp or heterogeneous_gnn.")

    # Create a validation dataloader
    valLoader: DataLoader = DataLoader(predict_dataset, batch_size=100, shuffle=False, num_workers=15)

    # Predict with the model
    pred = torch.zeros((0, 4))
    labels = torch.zeros((0, 4))
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
                pred_batch = torch.reshape(out_raw.squeeze(), (batch.batch_size, 4))

                # Get the labels
                labels_batch = torch.reshape(batch.y, (batch.batch_size, 4))

                # Append to the previously collected data
                pred = torch.cat((pred, pred_batch), dim=0)
                labels = torch.cat((labels, labels_batch), dim=0)

                if pred.shape[0] >= num_to_vis:
                    break

    return pred[0:num_to_vis], labels[0:num_to_vis]

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
    data_metadata = None
    if model_type == 'heterogeneous_gnn':
        if isinstance(train_dataset.dataset, torch.utils.data.ConcatDataset):
            data_metadata = train_dataset.dataset.datasets[0].dataset.get_data_metadata()
        elif isinstance(train_dataset.dataset, torch.utils.data.Dataset):
            data_metadata = train_dataset.dataset.get_data_metadata()

    # Set appropriate settings for testing mode
    limit_train_batches = None
    limit_val_batches = None
    limit_test_batches = None
    deterministic = False
    if testing_mode:
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
                                         num_workers=30,
                                         persistent_workers=True)
    valLoader: DataLoader = DataLoader(val_dataset,
                                       batch_size=batch_size,
                                       shuffle=False,
                                       num_workers=30,
                                       persistent_workers=True)
    testLoader: DataLoader = DataLoader(test_dataset,
                                        batch_size=batch_size,
                                        shuffle=False,
                                        num_workers=30)

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
    elif model_type == 'heterogeneous_gnn':
        lightning_model = Heterogeneous_GNN_Lightning(
            hidden_channels=hidden_size,
            edge_dim=dummy_batch['base', 'connect',
                                 'joint'].edge_attr.size()[1],
            num_layers=num_layers,
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
        filename='{epoch}-{' + monitor + ':.5f}',
        save_top_k=5,
        mode='min',
        monitor=monitor)
    last_model_callback = ModelCheckpoint(
        dirpath=path_to_save,
        filename='{epoch}-{' + monitor + ':.5f}',
        save_top_k=1,
        mode='max',
        monitor="epoch")

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
        # profiler="simple",
        max_epochs=epochs,
        limit_train_batches=limit_train_batches,
        limit_val_batches=limit_val_batches,
        limit_test_batches=limit_test_batches,
        check_val_every_n_epoch=1,
        enable_progress_bar=True,
        logger=wandb_logger,
        callbacks=[checkpoint_callback, last_model_callback])
    trainer.fit(lightning_model, trainLoader, valLoader)
    trainer.test(lightning_model, dataloaders=testLoader)

    # Return the path to the trained checkpoint
    return path_to_save
