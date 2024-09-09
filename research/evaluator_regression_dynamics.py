from pathlib import Path
from mi_hgnn.lightning_py.gnnLightning import train_model, evaluate_model
import torch
from mi_hgnn.datasets_py.quadSDKDataset import *
from mi_hgnn.visualization import visualize_model_outputs_regression

def main():
    path_to_urdf = Path('urdf_files', 'A1-Quad', 'a1_pruned.urdf').absolute()
    path_to_urdf_dynamics = Path('urdf_files', 'A1-Quad', 'a1.urdf').absolute()

    # Define model type
    model_type = 'dynamics'
    history_length = 1
    normalize = False

    # Initalize the datasets
    path_to_A1_1_0 = Path(Path('.').parent, 'datasets', 'QuadSDK-A1Speed1.0_DELETEME').absolute()
    a1_speed_1_0 = QuadSDKDataset_A1Speed1_0(path_to_A1_1_0, path_to_urdf, 'package://a1_description/', '', 
                                                        model_type, history_length , normalize, 
                                                        urdf_path_dynamics=path_to_urdf_dynamics)
    test_dataset = torch.utils.data.Subset(a1_speed_1_0, np.arange(0, a1_speed_1_0.__len__()))

    # Evaluate the model
    pred, labels, mse, rmse, l1 = evaluate_model(None, test_dataset)

    # Print the results
    print("Mean Squared Error: ", mse)
    print("Root Mean Squared Error: ", rmse)
    print("L1 Loss: ", l1)
    
    # Visualize the results
    s_i = 1000
    l = 1000
    visualize_model_outputs_regression(pred[s_i:s_i+l], labels[s_i:s_i+l])

if __name__ == '__main__':
     main()