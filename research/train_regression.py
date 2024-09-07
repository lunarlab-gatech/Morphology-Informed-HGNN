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
    history_length = 1
    normalize = False
    num_layers = 8
    hidden_size = 128

    # Initalize the datasets
    path_to_urdf = Path('urdf_files', 'Go2-Quad', 'go2.urdf').absolute()
    dataset_flat_0_5_50 = QuadSDKDataset_Go2_Flat_Speed0_5_Mu_50(Path(Path('.').parent, 'datasets', 'QuadSDK-Go2-Flat-0.5-Mu50').absolute(), 
            path_to_urdf, 'package://go2_description/', '', model_type, history_length, normalize=normalize)
    train_and_test_dataset = torch.utils.data.Subset(dataset_flat_0_5_50, np.arange(0, dataset_flat_0_5_50.__len__()))

    # Train the model
    train_model(train_and_test_dataset, train_and_test_dataset, train_and_test_dataset, normalize, 
                num_layers=num_layers, hidden_size=hidden_size, logger_project_name="delete_me", 
                batch_size=30, regression=True, lr=0.0001, epochs=49, seed=0, devices=1, early_stopping=True,
                testing_mode=True)

if __name__ == '__main__':
     main()