import torch
from torch import optim, nn
import lightning as L
from torch_geometric.nn.models import GCN, GraphSAGE
from torch_geometric.nn import to_hetero
from lightning.pytorch.loggers import WandbLogger
from .datasets import CerberusStreetDataset, CerberusTrackDataset, CerberusDataset, Go1SimulatedDataset
from torch_geometric.loader import DataLoader
from lightning.pytorch.callbacks import ModelCheckpoint
from pathlib import Path
import names
import matplotlib.pyplot as plt


class Base_Lightning(L.LightningModule):
    """
    Define training, validation, test, and prediction 
    steps used by all models, in addition to the 
    optimizer.
    """

    def log_losses(self, batch, mse_loss, y, y_pred, step_name: str):
        self.log(step_name + "_MSE_loss",
                 mse_loss,
                 batch_size=batch.batch_size)
        self.log(step_name + "_RMSE_loss",
                 torch.sqrt(mse_loss),
                 batch_size=batch.batch_size)
        self.log(step_name + "_L1_loss",
                 nn.functional.l1_loss(y, y_pred),
                 batch_size=batch.batch_size)

        # Log losses per individual leg
        for i in range(0, 4):
            y_leg = y[:, i]
            y_pred_leg = y_pred[:, i]
            leg_mse_loss = nn.functional.mse_loss(y_leg, y_pred_leg)
            self.log(step_name + "_MSE_loss:leg_" + str(i),
                     leg_mse_loss,
                     batch_size=batch.batch_size)
            self.log(step_name + "_RMSE_loss:leg_" + str(i),
                     torch.sqrt(leg_mse_loss),
                     batch_size=batch.batch_size)
            self.log(step_name + "_L1_loss:leg_" + str(i),
                     nn.functional.l1_loss(y_leg, y_pred_leg),
                     batch_size=batch.batch_size)

    def training_step(self, batch, batch_idx):
        mse_loss, y, y_pred = self.step_helper_function(batch, batch_idx)
        self.log_losses(batch, mse_loss, y, y_pred, "train")
        return mse_loss

    def validation_step(self, batch, batch_idx):
        mse_loss, y, y_pred = self.step_helper_function(batch, batch_idx)
        self.log_losses(batch, mse_loss, y, y_pred, "val")

    def test_step(self, batch, batch_idx):
        mse_loss, y, y_pred = self.step_helper_function(batch, batch_idx)
        self.log_losses(batch, mse_loss, y, y_pred, "test")

    def predict_step(self, batch, batch_idx):
        mse_loss, y, y_pred = self.step_helper_function(batch, batch_idx)
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


class GCN_Lightning(Base_Lightning):

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
        self.gcn_model = GCN(in_channels=num_node_features,
                             hidden_channels=hidden_channels,
                             num_layers=num_layers,
                             out_channels=1)
        self.y_indices = y_indices
        self.save_hyperparameters()

    def step_helper_function(self, batch, batch_idx):
        # Get the raw output
        out_raw = self.gcn_model(x=batch.x,
                                 edge_index=batch.edge_index).squeeze()

        # Reshape so that we have a tensor of (batch_size, num_nodes)
        out_nodes_by_batch = torch.reshape(
            out_raw,
            (batch.batch_size, int(batch.x.shape[0] / batch.batch_size)))

        # Get the outputs from the foot nodes
        truth_tensors = []
        for index in self.y_indices:
            truth_tensors.append(out_nodes_by_batch[:, index])
        y_pred = torch.stack(truth_tensors).swapaxes(0, 1)

        # Calculate loss
        y = torch.reshape(batch.y, (batch.batch_size, len(self.y_indices)))
        loss = nn.functional.mse_loss(y, y_pred)
        return loss, y, y_pred


class Heterogeneous_GNN_Lightning(Base_Lightning):

    def __init__(self, num_node_features, hidden_channels, num_layers,
                 y_indices, data_metadata, dummy_batch):
        """
        Constructor for Heterogeneous GNN.

        Parameters:
            num_node_features (int) - The number of features each
                node embedding has.
            hidden_channels (int) - The hidden size.
            num_layers (int) - The number of layers in the model.
            y_indices (list[int]) - The indices of the output
                that should match the ground truch labels provided.
                All other node outputs of the GNN are ignored.
            data_metadata (tuple) - See https://pytorch-geometric.readthedocs.io/en/latest/modules/nn.html?          highlight=to_hetero#torch_geometric.nn.to_hetero_transformer.to_hetero for details.
            dummy_batch - Used to initialize the lazy modules.
        """
        super().__init__()
        self.sage_model = GraphSAGE(in_channels=num_node_features,
                                    hidden_channels=hidden_channels,
                                    num_layers=num_layers,
                                    out_channels=1)
        self.sage_model = to_hetero(self.sage_model, data_metadata)
        self.y_indices = y_indices

        # # Initialize lazy modules
        with torch.no_grad():
            self.sage_model(x=dummy_batch.x_dict,
                            edge_index=dummy_batch.edge_index_dict)
        self.save_hyperparameters()

    def step_helper_function(self, batch, batch_idx):
        # Get the raw foot output
        out_raw = self.sage_model(x=batch.x_dict,
                                  edge_index=batch.edge_index_dict)['foot']

        # Reshape so that we have a tensor of (batch_size, num_foot_nodes)
        out_nodes_by_batch = torch.reshape(
            out_raw, (batch.batch_size,
                      int(batch['foot'].x.shape[0] / batch.batch_size)))

        # Get the outputs from the foot nodes
        truth_tensors = []
        for index in self.y_indices:
            truth_tensors.append(out_nodes_by_batch[:, index])
        y_pred = torch.stack(truth_tensors).swapaxes(0, 1)

        # Calculate loss
        y = torch.reshape(batch.y, (batch.batch_size, len(self.y_indices)))
        loss = nn.functional.mse_loss(y_pred, y)
        return loss, y, y_pred


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
        elif num_layers == 1:
            modules.append(nn.Linear(in_channels, 4))
            modules.append(nn.ReLU())
        elif num_layers == 2:
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
        return loss, y, y_pred


def display_on_axes(axes, estimated, ground_truth, title):
    """
    Simple function that displays grounth truth and estimated
    information on a Matplotlib.pyplot Axes.
    """
    axes.plot(ground_truth, label="Ground Truth", linestyle='-.')
    axes.plot(estimated, label="Estimated")
    axes.legend()
    axes.set_title(title)

def evaluate_model_and_visualize(model_type: str, path_to_checkpoint: Path, 
                            predict_dataset, subset_to_visualize: tuple[int], path_to_file: Path = None):

    # Initialize the model
    model = None
    if model_type is 'gnn':
        model = GCN_Lightning.load_from_checkpoint(str(path_to_checkpoint))
        model.num_nodes = predict_dataset.URDF.get_num_nodes()
        model.y_indices = predict_dataset.get_ground_truth_label_indices()
    elif model_type is 'mlp':
        model = MLP_Lightning.load_from_checkpoint(str(path_to_checkpoint))
    else:
        raise ValueError("model_type must be gnn or mlp.")

    # Create a validation dataloader
    valLoader: DataLoader = DataLoader(predict_dataset, batch_size=100, 
                                       shuffle=False, num_workers=15)

    # Setup four graphs
    fig, axes = plt.subplots(4, figsize=[20, 10])
    fig.suptitle('Foot Estimated Forces vs. Ground Truth')

    # Validate with the model
    trainer = L.Trainer()
    predictions_result = trainer.predict(model, valLoader)
    pred = torch.zeros((0, 4))
    labels = torch.zeros((0, 4))
    for batch_result in predictions_result:
        pred = torch.cat((pred, batch_result[0]), dim=0)
        labels = torch.cat((labels, batch_result[1]), dim=0)

    # Only use the specific subset chosen
    pred = pred.numpy()[subset_to_visualize[0]:subset_to_visualize[1]+1]
    labels = labels.numpy()[subset_to_visualize[0]:subset_to_visualize[1]+1]

    # Display the results
    titles = [
        "Front Left Foot Forces", "Front Right Foot Forces",
        "Rear Left Foot Forces", "Rear Right Foot Forces"
    ]
    for i in range(0, 4):
        display_on_axes(axes[i], pred[:, i], labels[:, i], titles[i])

    # Show the figure
    print(path_to_file)
    if path_to_file is not None:
        plt.savefig(path_to_file)
    plt.show()


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
    model_type = 'gnn'

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
    train_model(train_dataset, val_dataset, test_dataset, model_type,
                go1_sim_dataset.get_ground_truth_label_indices(),
                None)


def train_model(train_dataset, val_dataset, test_dataset, model_type: str,
                ground_truth_label_indices, data_metadata):

    # Set batch size
    batch_size = 100
    hidden_channels = 256
    num_layers = 8

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
    if model_type == 'gnn':
        lightning_model = GCN_Lightning(train_dataset[0].x.shape[1],
                                        hidden_channels, num_layers,
                                        ground_truth_label_indices)
    elif model_type == 'mlp':
        lightning_model = MLP_Lightning(24, hidden_channels, num_layers,
                                        batch_size)
    elif model_type == 'heterogeneous_gnn':
        lightning_model = Heterogeneous_GNN_Lightning(
            -1, hidden_channels, num_layers, ground_truth_label_indices,
            data_metadata, dummy_batch)
    else:
        raise ValueError("Invalid model type.")

    # Create Logger
    run_name = model_type + "-" + names.get_first_name(
    ) + "-" + names.get_last_name()
    wandb_logger = WandbLogger(project="grfgnn-Version2", name=run_name)
    wandb_logger.watch(lightning_model, log="all")

    # Set model parameters
    wandb_logger.experiment.config["batch_size"] = batch_size
    path_to_save = str(Path("models", wandb_logger.experiment.name))

    # Set up precise checkpointing
    checkpoint_callback = ModelCheckpoint(dirpath=path_to_save,
                                          filename='{epoch}-{val_MSE_loss:.2f}',
                                          save_top_k=5,
                                          monitor="val_MSE_loss")

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
        # limit_train_batches=10,
        # limit_val_batches=5,
        # limit_test_batches=5,
        check_val_every_n_epoch=1,
        enable_progress_bar=True,
        logger=wandb_logger,
        callbacks=[checkpoint_callback])
    trainer.fit(lightning_model, trainLoader, valLoader)
    trainer.test(lightning_model, dataloaders=testLoader)
