from pathlib import Path
from mi_hgnn.lightning_py.gnnLightning import train_model, evaluate_model
import torch
from mi_hgnn.datasets_py.quadSDKDataset import *
from mi_hgnn.visualization import visualize_model_outputs_regression

def main():
    # ================================= CHANGE THESE ===================================
    model_type = 'heterogeneous_gnn'
    # ==================================================================================

    # Define model information
    history_length = 150
    normalize = False
    num_layers = 8
    hidden_size = 128

    # Set up the urdf paths
    path_to_urdf = Path('urdf_files', 'A1-Quad', 'a1_pruned.urdf').absolute()
    path_to_urdf_dynamics = Path('urdf_files', 'A1-Quad', 'a1.urdf').absolute()

    # Initalize the Train datasets
    bravo = QuadSDKDataset_A1_Bravo(Path(Path('.').parent, 'datasets', 'QuadSDK-A1-Bravo').absolute(), path_to_urdf, 
                'package://a1_description/', '', model_type, history_length, normalize, path_to_urdf_dynamics)
    charlie = QuadSDKDataset_A1_Charlie(Path(Path('.').parent, 'datasets', 'QuadSDK-A1-Charlie').absolute(), path_to_urdf, 
                'package://a1_description/', '', model_type, history_length, normalize, path_to_urdf_dynamics)
    echo = QuadSDKDataset_A1_Echo(Path(Path('.').parent, 'datasets', 'QuadSDK-A1-Echo').absolute(), path_to_urdf, 
                'package://a1_description/', '', model_type, history_length, normalize, path_to_urdf_dynamics)
    foxtrot = QuadSDKDataset_A1_Foxtrot(Path(Path('.').parent, 'datasets', 'QuadSDK-A1-Foxtrot').absolute(), path_to_urdf, 
                'package://a1_description/', '', model_type, history_length, normalize, path_to_urdf_dynamics)
    juliett = QuadSDKDataset_A1_Juliett(Path(Path('.').parent, 'datasets', 'QuadSDK-A1-Juliett').absolute(), path_to_urdf, 
                'package://a1_description/', '', model_type, history_length, normalize, path_to_urdf_dynamics)
    kilo = QuadSDKDataset_A1_Kilo(Path(Path('.').parent, 'datasets', 'QuadSDK-A1-Kilo').absolute(), path_to_urdf, 
                'package://a1_description/', '', model_type, history_length, normalize, path_to_urdf_dynamics)
    mike = QuadSDKDataset_A1_Mike(Path(Path('.').parent, 'datasets', 'QuadSDK-A1-Mike').absolute(), path_to_urdf, 
                'package://a1_description/', '', model_type, history_length, normalize, path_to_urdf_dynamics)
    november = QuadSDKDataset_A1_November(Path(Path('.').parent, 'datasets', 'QuadSDK-A1-November').absolute(), path_to_urdf, 
                'package://a1_description/', '', model_type, history_length, normalize, path_to_urdf_dynamics)
    
    # Define train and val sets
    train_val_datasets = [bravo, charlie, echo, foxtrot, juliett, kilo, mike, november]

    # Take first 85% for training, and last 15% for validation
    # Also remove the last entries, as dynamics models can't use last entry due to derivative calculation
    train_subsets = []
    val_subsets = []
    for dataset in train_val_datasets:
        data_len_minus_1 = dataset.__len__() - 1
        split_index = int(np.round(data_len_minus_1 * 0.85)) # When value has .5, round to nearest-even
        train_subsets.append(torch.utils.data.Subset(dataset, np.arange(0, split_index)))
        val_subsets.append(torch.utils.data.Subset(dataset, np.arange(split_index, data_len_minus_1)))
    train_dataset = torch.utils.data.ConcatDataset(train_subsets)
    val_dataset = torch.utils.data.ConcatDataset(val_subsets)
    
    # Convert them to subsets
    train_dataset = torch.utils.data.Subset(train_dataset, np.arange(0, train_dataset.__len__()))
    val_dataset = torch.utils.data.Subset(val_dataset, np.arange(0, val_dataset.__len__()))
    
    # Train the model (evaluate later, so no test set)
    train_model(train_dataset, val_dataset, None, normalize, 
                num_layers=num_layers, hidden_size=hidden_size, logger_project_name="regression_experiment", 
                batch_size=30, regression=True, lr=0.0001, epochs=30, seed=0, devices=1, early_stopping=True,
                disable_test=True)

if __name__ == '__main__':
     main()