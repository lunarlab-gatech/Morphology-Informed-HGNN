from pathlib import Path
from grfgnn.lightning_py.gnnLightning import evaluate_model
from grfgnn import QuadSDKDataset_A1Speed0_5, QuadSDKDataset_A1Speed1_0, QuadSDKDataset_A1Speed1_5FlippedOver
from grfgnn.visualization import visualize_model_outputs_classification
from grfgnn.lightning_py.gnnLightning import Base_Lightning
import torch
from torch.utils.data import Subset
import grfgnn.datasets_py.LinTzuYaunDataset as linData
import numpy as np
import torchmetrics


def main():
    # ================================= CHANGE THESE ===================================
    path_to_checkpoint = "/home/dbutterfield3/Research/state-estimation-gnn/models/ancient-salad-5/epoch=29-val_CE_loss=0.34249.ckpt"
    model_type = 'heterogeneous_gnn'
    num_entries_to_eval = 100
    # ==================================================================================

    # Set parameters
    history_length = 150
    path_to_urdf = Path('urdf_files', 'MiniCheetah', 'miniCheetah.urdf').absolute()

    # Initialize the Testing datasets
    air_jumping_gait = linData.LinTzuYaunDataset_air_jumping_gait(
        Path(Path('.').parent, 'datasets', 'LinTzuYaun-AJG').absolute(), path_to_urdf, 'package://yobotics_description/', 'mini-cheetah-gazebo-urdf/yobo_model/yobotics_description', model_type, history_length, normalize=False)
    concrete_pronking = linData.LinTzuYaunDataset_concrete_pronking(
        Path(Path('.').parent, 'datasets', 'LinTzuYaun-CP').absolute(), path_to_urdf, 'package://yobotics_description/', 'mini-cheetah-gazebo-urdf/yobo_model/yobotics_description', model_type, history_length, normalize=False)
    concrete_right_circle = linData.LinTzuYaunDataset_concrete_right_circle(
        Path(Path('.').parent, 'datasets', 'LinTzuYaun-CRC').absolute(), path_to_urdf, 'package://yobotics_description/', 'mini-cheetah-gazebo-urdf/yobo_model/yobotics_description', model_type, history_length, normalize=False)
    forest = linData.LinTzuYaunDataset_forest(
        Path(Path('.').parent, 'datasets', 'LinTzuYaun-F').absolute(), path_to_urdf, 'package://yobotics_description/', 'mini-cheetah-gazebo-urdf/yobo_model/yobotics_description', model_type, history_length, normalize=False)
    small_pebble = linData.LinTzuYaunDataset_small_pebble(
        Path(Path('.').parent, 'datasets', 'LinTzuYaun-SP').absolute(), path_to_urdf, 'package://yobotics_description/', 'mini-cheetah-gazebo-urdf/yobo_model/yobotics_description', model_type, history_length, normalize=False)
    test_dataset = torch.utils.data.ConcatDataset([air_jumping_gait, concrete_pronking, concrete_right_circle, forest, small_pebble])

    # Convert them to subsets
    test_dataset = torch.utils.data.Subset(test_dataset, np.arange(0, test_dataset.__len__()))

    # Evaluate with model
    pred, labels = evaluate_model(path_to_checkpoint, test_dataset, num_entries_to_eval)

    # Output the corresponding results
    metric_acc = torchmetrics.Accuracy(task="multiclass", num_classes=16)
    y_pred_16, y_16 = Base_Lightning.classification_conversion_16_class(None, pred, labels)
    print("Accuracy: ", metric_acc(torch.argmax(y_pred_16, dim=1), y_16.squeeze()))
    visualize_model_outputs_classification(pred, labels, str(path_to_checkpoint) + ".pdf", 100)

if __name__ == "__main__":
    main()
