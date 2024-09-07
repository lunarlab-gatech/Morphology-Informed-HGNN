from pathlib import Path
from mi_hgnn.lightning_py.gnnLightning import train_model, evaluate_model
import torch
from mi_hgnn.datasets_py.quadSDKDataset import *
from mi_hgnn.visualization import visualize_model_outputs_regression

def main():
    path_to_urdf = Path('urdf_files', 'Go2-Quad', 'go2.urdf').absolute()

    # Define model type
    model_type = 'dynamics'
    history_length = 1
    normalize = False

    # Initalize the datasets
    dataset_flat_0_5_50 = QuadSDKDataset_Go2_Flat_Speed0_5_Mu_50(Path(Path('.').parent, 'datasets', 'QuadSDK-Go2-Flat-0.5-Mu50').absolute(), 
            path_to_urdf, 'package://go2_description/', '', model_type, history_length, normalize=normalize)
    test_dataset = torch.utils.data.Subset(dataset_flat_0_5_50, np.arange(0, dataset_flat_0_5_50.__len__()))

    # Evaluate the model
    pred, labels, mse, rmse, l1 = evaluate_model(None, test_dataset)

    # Print the results
    print("Mean Squared Error: ", mse)
    print("Root Mean Squared Error: ", rmse)
    print("L1 Loss: ", l1)
    
    visualize_model_outputs_regression(pred[0:1000], labels[0:1000])

if __name__ == '__main__':
     main()