import torch
from torch import optim, nn
import lightning as L
from lightning.pytorch import seed_everything
from lightning.pytorch.loggers import WandbLogger
from torch_geometric.loader import DataLoader
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from pathlib import Path
import names
from torch.utils.data import Subset
import numpy as np
import torchmetrics
import torchmetrics.classification
import pinocchio as pin
from .customMetrics import CrossEntropyLossMetric, BinaryF1Score
from .hgnn import GRF_HGNN
from torch_geometric.profile import count_parameters
from ..datasets_py.flexibleDataset import FlexibleDataset

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

        # ====== Setup the metrics ======
        self.metric_mse: torchmetrics.MeanSquaredError = torchmetrics.regression.MeanSquaredError(squared=True)
        self.metric_rmse: torchmetrics.MeanSquaredError = torchmetrics.regression.MeanSquaredError(squared=False)
        self.metric_l1: torchmetrics.MeanAbsoluteError = torchmetrics.regression.MeanAbsoluteError()

        self.metric_ce = CrossEntropyLossMetric()
        self.metric_acc = torchmetrics.Accuracy(task="multiclass", num_classes=16)
        self.metric_f1_leg0 = BinaryF1Score()
        self.metric_f1_leg1 = BinaryF1Score()
        self.metric_f1_leg2 = BinaryF1Score()
        self.metric_f1_leg3 = BinaryF1Score()

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
    def calculate_losses_step(self, y: torch.Tensor, y_pred: torch.Tensor):
        if self.regression:
            y = y.flatten()
            y_pred = y_pred.flatten()
            self.mse_loss = self.metric_mse(y_pred, y)
            self.rmse_loss = self.metric_rmse(y_pred, y)
            self.l1_loss = self.metric_l1(y_pred, y)
        else:
            batch_size = y_pred.shape[0]

            # Calculate useful values
            y_pred_per_foot, y_pred_per_foot_prob, y_pred_per_foot_prob_only_1 = \
                    self.classification_calculate_useful_values(y_pred, batch_size)

            # Calculate CE_loss
            self.ce_loss = self.metric_ce(y_pred_per_foot, y.long().flatten())

            # Calculate 16 class accuracy
            y_pred_16, y_16 = self.classification_conversion_16_class(y_pred_per_foot_prob_only_1, y)
            y_16 = y_16.squeeze(dim=1)
            self.acc = self.metric_acc(torch.argmax(y_pred_16, dim=1), y_16)

            # Calculate binary class predictions for f1-scores
            y_pred_2 = torch.reshape(torch.argmax(y_pred_per_foot_prob, dim=1), (batch_size, 4))
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
    # NOTE: These methods have not been fully tested. Use at 
    # your own risk.

    def on_predict_start(self):
        self.reset_all_metrics()

    def predict_step(self, batch, batch_idx):
        """
        Returns the predicted values from the model given a specific batch. 
        Currently only implemented for regression.

        Note, the HGNN models (Both regression and classification) directly
        predict with model, instead of using this built-in method.

        Returns:
            y (torch.Tensor) - Ground Truth labels per foot (GRF labels for
              regression, 16 class contact labels for classifiction)
            y_pred (torch.Tensor) - Predicted outputs (GRF labels per foot 
                for regression, 16 class predictions for classifications)
        """
        y, y_pred = self.step_helper_function(batch)
        self.calculate_losses_step(y, y_pred)

        if self.regression:
            return y, y_pred # GRFs
        else:
            raise NotImplementedError("This prediction method is not fully tested for classification.")
        
            y_pred_per_foot, y_pred_per_foot_prob, y_pred_per_foot_prob_only_1 = \
                    self.classification_calculate_useful_values(y_pred, y_pred.shape[0])
            y_pred_16, y_16 = self.classification_conversion_16_class(y_pred_per_foot_prob_only_1, y)
            y_16 = y_16.squeeze(dim=1)
            return y_16, torch.argmax(y_pred_16, dim=1) # 16 class
        
    def on_predict_end(self):
        self.calculate_losses_epoch()

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
            y: The ground truth labels with the shape (batch_size, 4).
            y_pred: The predicted model output. If self.regression, these are 
                just GRF values with the shape (batch_size, 4). If not 
                self.regression, then these are contact probabilty logits,
                two per foot, with shape (batch_size, 4). Foot order matches
                order of URDF file, and logit assumes first value is logit of
                no contact, and second value is logit of contact.

        """
        raise NotImplementedError

    def classification_calculate_useful_values(self, y_pred, batch_size):
        """
        Helper method that calculates useful values for us:

        Returns:
            y_pred_per_foot (torch.Tensor): Contains contact probability logits per foot, 
                in the shape (batch_size * 4, 2). In dimension 1, first value represents
                logit for no/unstable contact, and second value respresents logit for
                stable contact.
            y_pred_per_foot_prob (torch.Tensor): Same as above, but contains probablity 
                values instead of logits.
            y_pred_per_foot_prob_only_1 (torch.Tensor): Contains probabilities of stable
                contact for each foot, in the shape (batch_size, 4). Note that there are no
                probabilities of unstable contact.
        """
        y_pred_per_foot = torch.reshape(y_pred, (batch_size * 4, 2))
        y_pred_per_foot_prob = torch.nn.functional.softmax(y_pred_per_foot, dim=1)
        y_pred_per_foot_prob_only_1 = torch.reshape(y_pred_per_foot_prob[:,1], (batch_size, 4))

        return y_pred_per_foot, y_pred_per_foot_prob, y_pred_per_foot_prob_only_1

    def classification_conversion_16_class(self, y_pred_per_foot_prob_only_1: torch.Tensor, y: torch.Tensor):
        """
        Convert the y labels from individual foot contact classes into a single 
        class out of 16 options. In other words, convert y from (batch_size, 4) to
        (batch_size, 1).

        Also, convert the y_pred_per_foot_prob_only_1 from individual foot contact 
        probabilities into a 16 class probability which comprises the four states. 
        In other words, convert y_pred from (batch_size, 4) to (batch_size, 16).

        Parameters:
            y_pred_per_foot_prob_only_1(torch.Tensor): A Tensor of shape 
                (batch_size, 4), containing the probability values of stable contact 
                for each individual foot.
            y (torch.Tensor): A Tensor of shape (batch_size, 4).
        """

        # Convert y labels from four sets of 2 classes to one set of 16 classes
        y_np = y.cpu().numpy()
        y_new = np.zeros((y.shape[0], 1))
        for i in range(0, y.shape[0]):
            y_new[i] = y_np[i,0] * 8 + y_np[i,1] * 4 + y_np[i,2] * 2 + y_np[i,3]
        y_new = torch.tensor(y_new, dtype=int)

        # Convert y_pred from two class predictions per foot to a single 16 class prediction
        y_pred = y_pred_per_foot_prob_only_1
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
    
class Test_Lighting_Model(Base_Lightning):
    """
    This lightning model is used for testing purposes.
    """
    def __init__(self):
        super().__init__("adam", 0.001, regression=True)

    def step_helper_function(self, batch):
        x, y = batch
        y_pred = y * 3
        return y, y_pred


class MLP_Lightning(Base_Lightning):

    def __init__(self, in_channels: int, hidden_channels: int,
                 out_channels: int, num_layers: int,
                 batch_size: int, optimizer: str = "adam", lr: float = 0.003,
                 regression: bool = True, activation_fn = nn.ReLU()):
        """
        Constructor for MLP_Lightning class. Pytorch Lightning
        wrapper around the Pytorch Torchvision MLP class.

        Parameters:
            in_channels (int): Number of input parameters to the MLP.
            hidden_channels (int): The hidden size.
            out_channels (int): The number of outputs from the MLP.
            num_layers (int): The number of layers in the model.
            batch_size (int): The size of the batches from the dataloaders.
            optimizer (str): String name of the optimizer that should
                be used.
            lr (float): The learning rate used by the model.
            regression (bool): True if the problem is regression, false if 
                classification. Mainly for tracking model usage using W&B.
            activation_fn (class): The activation function used between the layers.
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
        else:
            modules.append(nn.Linear(in_channels, hidden_channels))
            modules.append(activation_fn)
            for i in range(0, num_layers - 2):
                modules.append(nn.Linear(hidden_channels, hidden_channels))
                modules.append(activation_fn)
            modules.append(nn.Linear(hidden_channels, out_channels))

        self.mlp_model = nn.Sequential(*modules)
        self.save_hyperparameters()

    def step_helper_function(self, batch):
        x, y = batch
        y_pred = self.mlp_model(x)
        return y, y_pred

class Heterogeneous_GNN_Lightning(Base_Lightning):

    def __init__(self, hidden_channels: int, num_layers: int, data_metadata,
                 dummy_batch, optimizer: str = "adam", lr: float = 0.003,
                 regression: bool = True, activation_fn = nn.ReLU()):
        """
        Constructor for Heterogeneous GNN.

        Parameters:
            dummy_batch: Used to initialize the lazy modules.
            optimizer (str): String name of the optimizer that should
                be used.
            lr (float): The learning rate used by the model.

            See hgnn.py for information on remaining parameters.
        """
        super().__init__(optimizer, lr, regression)
        self.model = GRF_HGNN(hidden_channels=hidden_channels,
                              num_layers=num_layers,
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
        batch_size = None
        if hasattr(batch, "batch_size"):
            batch_size = batch.batch_size
        else:
            batch_size = 1
        y_pred = torch.reshape(out_raw.squeeze(), (batch_size, self.model.out_channels_per_foot * 4))

        # Get the labels
        y = torch.reshape(batch.y, (batch_size, 4))
        return y, y_pred
    
class Full_Dynamics_Model_Lightning(Base_Lightning):

    def __init__(self, urdf_model_path: Path, urdf_dir: Path, 
                 pin_to_urdf_joint_mapping: np.ndarray,
                 pin_to_urdf_foot_mapping: np.ndarray):
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
            pin_to_urdf_joint_mapping (np.ndarray) - An array that maps Pin joint 
                order to URDF joint order, of shape [12].
            pin_to_urdf_foot_mapping (np.ndarray) - An array that maps Pin foot 
                order to URDF foot order, of shape [4].
        """

        # Note we set optimizer and lr, but they are unused here
        # as we don't instantiate any underlying model.
        super().__init__("adam", 0.001, regression=True)
        self.regression = True
        self.joint_mapping = pin_to_urdf_joint_mapping
        self.foot_mapping = pin_to_urdf_foot_mapping
        self.save_hyperparameters()

        # Build the pinnochio model
        self.model, self.collision_model, self.visual_model = pin.buildModelsFromUrdf(
            str(urdf_model_path), str(urdf_dir), pin.JointModelFreeFlyer(), verbose=True
        )
        self.data = self.model.createData()

        # Setup feet frames ids in Pinnochio foot order
        self.feet_names = ["jtoe2", "jtoe3", "jtoe0", "jtoe1"]
        self.feet_ids = [self.model.getFrameId(n) for n in self.feet_names]

        # Get number of contact points
        self.ncontact = len(self.feet_names)

        # Setup a viewer to see if everything looks right
        self.viz = pin.visualize.MeshcatVisualizer(self.model, self.collision_model, self.visual_model)
        self.viz.initViewer(loadModel=True)

    def step_helper_function(self, batch):
        # Get the data from the batch
        q, vel, acc, tau, y = batch
        batch_size = y.shape[0]

        # Convert to numpy arrays
        q = q.cpu().numpy()
        vel = vel.cpu().numpy()
        acc = acc.cpu().numpy()
        tau = tau.cpu().numpy()

        # Set acceleration zero vector for drift calculation
        a0 = np.zeros(self.model.nv)

        # Setup array to hold predicted values
        y_pred = torch.zeros((batch_size, 4), dtype=torch.float64)

        # Evaluate on a sequence wise basis
        for i in range(0, batch_size):
            # Find Mass matrix
            # Why upper triangular?
            # In local or world frame? 
            # Documentation seems to imply world by default.
            M = pin.crba(self.model, self.data, q[i])

            # Compute dynamic drift -- Coriolis, centrifugal, gravity
            drift = pin.rnea(self.model, self.data, q[i], vel[i], a0)

            # Now, we need to find the contact Jacobians.
            # These are the Jacobians that relate the joint velocity to the velocity of each feet
            # Computed in the local coordinate system
            J_feet = [np.copy(pin.computeFrameJacobian(self.model, self.data, q[i], id, pin.LOCAL)) for id in self.feet_ids]

            # Extract the first three rows, probably to remove
            # the parts that connect joint velocity to foot angular velocity and just consider foot linear velocity
            J_feet_first_3_rows = [np.copy(J[:3, :]) for J in J_feet] 
            J_feet_T = np.zeros([18, 3 * self.ncontact])
            J_feet_T[:, :] = np.vstack(J_feet_first_3_rows).T

            # Contact forces at local coordinates (at each foot coordinate, thus Local Frame)
            contact_forces = np.linalg.pinv(J_feet_T) @ ((M @ acc[i]) + drift - tau[i])
            contact_forces_split = np.split(contact_forces, self.ncontact)

            # Compute the placement of each frame
            pin.framesForwardKinematics(self.model, self.data, q[i])
            # self.viz.display(q[i])
            # import time
            # time.sleep(0.1)

            # Convert Contact forces to World frame
            index = 0
            for force, foot_id in zip(contact_forces_split, self.feet_ids):
                force_transpose = np.array([[force[0]], [force[1]], [force[2]]])

                # Get the Foot to World Transform
                world_to_foot_SE3 = self.data.oMf[foot_id]
                foot_to_world_SE3 = world_to_foot_SE3.inverse()

                # Transform the force into the world frame
                world_frame_force = foot_to_world_SE3.rotation @ force_transpose

                # Save this prediction into the predicted array
                y_pred[i, index] = torch.tensor(-world_frame_force[2], dtype=torch.float64)

                # Increase index
                index += 1

        # Knowing that we can't have negative contact force, set all
        # negative values to zero.
        y_pred = torch.clamp(y_pred, min=0.0)

        # Put y_pred back on the correct device
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        y_pred = y_pred.to(device)

        # y and y_pred is in pin foot order, so map back to URDF foot order
        y = y[:, self.foot_mapping]
        y_pred = y_pred[:, self.foot_mapping]

        return y, y_pred


def evaluate_model(path_to_checkpoint: Path, predict_dataset: Subset,
                   enable_testing_mode: bool = False):
    """
    Runs the provided model on the corresponding dataset,
    and returns the predicted values and the ground truth values.

    Parameters:
        enable_testing_mode: Only used for test cases, don't enable
            unless you are writing test cases.

    Returns:
        pred - Predicted values
        labels - Ground Truth values
        * - Additional arguments that correspond to the metrics tracked
            during the evaluation.
    """

    # Set the dtype to be 64 by default
    torch.set_default_dtype(torch.float64)

    # Get the model type
    model_type = None
    dataset_raw: FlexibleDataset = None
    if isinstance(predict_dataset.dataset, torch.utils.data.ConcatDataset):
        dataset_raw = predict_dataset.dataset.datasets[0]
    elif isinstance(predict_dataset.dataset, torch.utils.data.Dataset):
        dataset_raw = predict_dataset.dataset
    model_type = dataset_raw.get_data_format()

    # Initialize the model
    model: Base_Lightning = None
    if enable_testing_mode:
        model = Test_Lighting_Model()
    elif model_type == 'mlp':
        model = MLP_Lightning.load_from_checkpoint(str(path_to_checkpoint))
    elif model_type == 'heterogeneous_gnn':
        model = Heterogeneous_GNN_Lightning.load_from_checkpoint(str(path_to_checkpoint))
    elif model_type == 'dynamics':
        urdf_path = None
        try:
            urdf_path = Path(dataset_raw.robotGraphFull.new_urdf_path)
        except:
            raise ValueError("urdf_path_dynamics needs to be passed to FlexibleDataset in order to use Dynamics model.")
        joint_mapping, foot_mapping = dataset_raw.pin_to_urdf_order_mapping()
        model = Full_Dynamics_Model_Lightning(str(urdf_path), urdf_path.parent.parent,
                                            joint_mapping, foot_mapping)
                                                
    else:
        raise ValueError("model_type must be mlp, heterogeneous_gnn, or dynamics.")
    model.eval()
    model.freeze()

    # Create a validation dataloader
    valLoader: DataLoader = DataLoader(predict_dataset, batch_size=100, shuffle=False, num_workers=15)

    # Predict with the model
    pred = torch.zeros((0, 4))
    labels = torch.zeros((0, 4))
    if (model_type == 'dynamics' or model_type == 'mlp') and model.regression:
        trainer = L.Trainer()
        predictions_result = trainer.predict(model, valLoader)
        for batch_result in predictions_result:
            labels = torch.cat((labels, batch_result[0]), dim=0)
            pred = torch.cat((pred, batch_result[1]), dim=0)
        
        # Return the results
        return pred, labels, model.mse_loss, model.rmse_loss, model.l1_loss

    elif model_type == 'heterogeneous_gnn':
        pred = torch.zeros((0))
        labels = torch.zeros((0))
        device = 'cpu'  # 'cuda' if torch.cuda.is_available() else
        model.model = model.model.to(device)
        with torch.no_grad():
            # Print visual output of prediction step
            total_batches = len(valLoader)
            batch_num = 0
            print("Prediction: ", batch_num, "/", total_batches, "\r", end="")

            # Predict with the model
            for batch in valLoader:
                labels_batch, y_pred = model.step_helper_function(batch)
                model.calculate_losses_step(labels_batch, y_pred)

                # If classification, convert to 16 class predictions and labels
                if not model.regression:
                    y_pred_per_foot, y_pred_per_foot_prob, y_pred_per_foot_prob_only_1 = \
                        model.classification_calculate_useful_values(y_pred, y_pred.shape[0])
                    y_pred_16, y_16 = model.classification_conversion_16_class(y_pred_per_foot_prob_only_1, labels_batch)
                    y_16 = y_16.squeeze(dim=1)
                    pred_batch = torch.argmax(y_pred_16, dim=1)
                    labels_batch = y_16
                else:
                    pred_batch = y_pred

                # Append to the previously collected data
                pred = torch.cat((pred, pred_batch), dim=0)
                labels = torch.cat((labels, labels_batch), dim=0)

                # Print current status
                batch_num += 1
                print("Prediction: ", batch_num, "/", total_batches, "\r", end="")

            model.calculate_losses_epoch()
        
        # Return the results
        if not model.regression:
            legs_avg_f1 = (model.f1_leg0 + model.f1_leg1 + model.f1_leg2 + model.f1_leg3) / 4.0
            return pred, labels, model.acc, model.f1_leg0, model.f1_leg1, model.f1_leg2, model.f1_leg3, legs_avg_f1
        else:
            return pred, labels, model.mse_loss, model.rmse_loss, model.l1_loss 
    
    else:
        problem_type = "regression" if model.regression else "classification"
        raise NotImplementedError("This combination of model_type (" + model_type + \
                                  ") and problem_type  (" + problem_type + ") is not \
                                  currently implemented for evaluation.")
  

def train_model(
        train_dataset: Subset,
        val_dataset: Subset,
        test_dataset: Subset,
        normalize: bool,  # Note, this is just so that we can log if the datasets were normalized.
        testing_mode: bool = False,
        disable_logger: bool = False,
        logger_project_name: str = None,
        batch_size: int = 100,
        num_layers: int = 8,
        optimizer: str = "adam",
        lr: float = 0.003,
        epochs: int = 30,
        hidden_size: int = 10,
        regression: bool = True,
        seed: int = 0,
        devices: int = 1,
        early_stopping: bool = False,
        disable_test: bool = False):
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
        train_data_format = train_dataset.dataset.datasets[
            0].dataset.get_data_format()
        val_data_format = val_dataset.dataset.datasets[
            0].dataset.get_data_format()
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
            data_metadata = train_dataset.dataset.datasets[
                0].dataset.get_data_metadata()
        elif isinstance(train_dataset.dataset, torch.utils.data.Dataset):
            data_metadata = train_dataset.dataset.get_data_metadata()

    # Set appropriate settings for testing mode
    limit_train_batches = None
    limit_val_batches = None
    limit_test_batches = None
    limit_predict_batches = None
    num_workers = 30
    persistent_workers = True
    if testing_mode:
        limit_train_batches = 10
        limit_val_batches = 5
        limit_test_batches = 5
        limit_predict_batches = limit_test_batches * batch_size
        num_workers = 1
        persistent_workers = False

    # Set the dtype to be 64 by default
    torch.set_default_dtype(torch.float64)

    # Create the dataloaders
    trainLoader: DataLoader = DataLoader(train_dataset,
                                         batch_size=batch_size,
                                         shuffle=True,
                                         num_workers=num_workers,
                                         persistent_workers=persistent_workers)
    valLoader: DataLoader = DataLoader(val_dataset,
                                       batch_size=batch_size,
                                       shuffle=False,
                                       num_workers=num_workers,
                                       persistent_workers=persistent_workers)
    testLoader = None
    if not disable_test:
        testLoader: DataLoader = DataLoader(test_dataset,
                                            batch_size=batch_size,
                                            shuffle=False,
                                            num_workers=num_workers)
    
    # Set a random seed (need to be before we get dummy_batch)
    seed_everything(seed, workers=True)

    # Get a dummy_batch
    dummy_batch = None
    for batch in trainLoader:
        dummy_batch = batch
        break

    # Create the model
    lightning_model = None
    model_parameters = None
    if model_type == 'mlp':

        # Determine the number of output channels
        out_channels = None
        if regression:
            out_channels = 4
        else:
            out_channels = 8

        # Create the model
        lightning_model = MLP_Lightning(
            in_channels=train_dataset[0][0].shape[0],
            hidden_channels=hidden_size,
            out_channels=out_channels,
            num_layers=num_layers,
            batch_size=batch_size,
            optimizer=optimizer,
            lr=lr,
            regression=regression)
        model_parameters = count_parameters(lightning_model.mlp_model)

    elif model_type == 'heterogeneous_gnn':
        lightning_model = Heterogeneous_GNN_Lightning(
            hidden_channels=hidden_size,
            num_layers=num_layers,
            data_metadata=data_metadata,
            dummy_batch=dummy_batch,
            optimizer=optimizer,
            lr=lr,
            regression=regression)
        model_parameters = count_parameters(lightning_model.model)
    else:
        raise ValueError("Invalid model type.")

    # Create Logger
    wandb_logger = False
    path_to_save = None
    if not disable_logger:
        if logger_project_name is None:
            raise ValueError(
                "Need to define \"logger_project_name\" if logger is enabled.")
        wandb_logger = WandbLogger(project=logger_project_name)
        wandb_logger.watch(lightning_model, log="all")
        wandb_logger.experiment.config["batch_size"] = batch_size
        wandb_logger.experiment.config["normalize"] = normalize
        wandb_logger.experiment.config["num_parameters"] = model_parameters
        wandb_logger.experiment.config["seed"] = seed
        path_to_save = str(Path("models", wandb_logger.experiment.name))
    else:
        path_to_save = str(
            Path("models", model_type + "_" + names.get_full_name()))

    # Set up precise checkpointing
    monitor = None
    if regression:
        monitor = "val_MSE_loss"
    else:
        monitor = "val_CE_loss"
    checkpoint_callback = ModelCheckpoint(dirpath=path_to_save,
                                          filename='{epoch}-{' + monitor +
                                          ':.5f}',
                                          save_top_k=5,
                                          mode='min',
                                          monitor=monitor)
    last_model_callback = ModelCheckpoint(dirpath=path_to_save,
                                          filename='{epoch}-{' + monitor +
                                          ':.5f}',
                                          save_top_k=1,
                                          mode='max',
                                          monitor="epoch")

    # Lower precision of operations for faster training
    torch.set_float32_matmul_precision("medium")

    # Setup early stopping mechanism to match MorphoSymm-Replication
    callbacks = [checkpoint_callback, last_model_callback]
    if early_stopping:
        callbacks.append(EarlyStopping(monitor=monitor, patience=10, mode='min'))

    # Train the model and test
    trainer = L.Trainer(
        default_root_dir=path_to_save,
        deterministic=True,  # Reproducability
        devices=devices,
        accelerator="auto",
        # profiler="simple",
        max_epochs=epochs,
        limit_train_batches=limit_train_batches,
        limit_val_batches=limit_val_batches,
        limit_test_batches=limit_test_batches,
        limit_predict_batches=limit_predict_batches,
        check_val_every_n_epoch=1,
        enable_progress_bar=True,
        logger=wandb_logger,
        callbacks=callbacks)
    trainer.fit(lightning_model, trainLoader, valLoader)
    if not disable_test:
        trainer.test(lightning_model, dataloaders=testLoader, verbose=True)

    # Return the path to the trained checkpoint
    return path_to_save
