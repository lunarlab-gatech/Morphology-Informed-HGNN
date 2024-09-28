import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from pathlib import Path
from mi_hgnn.lightning_py.gnnLightning import evaluate_model
import torch
from mi_hgnn.datasets_py.quadSDKDataset import *

def evaluation_example():

    # We'll pick the MLP and HGNN with lowest RMSE on Unseen All Dataset
    hgnn_path = 'models_regression/hgnns/wise-firebrand-23/epoch=27-val_MSE_loss=74.51168.ckpt'
    mlp_path = 'models_regression/mlps/fast-hill-18/epoch=29-val_MSE_loss=107.37196.ckpt'

    # Set up the urdf paths
    path_to_urdf = Path('urdf_files', 'A1-Quad', 'a1_pruned.urdf').absolute()
    path_to_urdf_dynamics = Path('urdf_files', 'A1-Quad', 'a1.urdf').absolute()

    # None of the models Normalize
    normalize = False

    # Evaluate Unseen All on Heterogenous GNN
    uniform = QuadSDKDataset_A1_Uniform(Path(Path('.').parent, 'datasets', 'QuadSDK-A1-Uniform').absolute(), path_to_urdf, 
                'package://a1_description/', '', 'heterogeneous_gnn', 150, normalize, path_to_urdf_dynamics)
    unseen_all_dataset = torch.utils.data.ConcatDataset([torch.utils.data.Subset(uniform, np.arange(0, uniform.__len__() - 1))])
    pred_hgnn, labels_hgnn, mse, rmse, l1 = evaluate_model(hgnn_path, torch.utils.data.Subset(unseen_all_dataset, np.arange(0, unseen_all_dataset.__len__())))

    # Evaluate Unseen All on MLP
    uniform = QuadSDKDataset_A1_Uniform(Path(Path('.').parent, 'datasets', 'QuadSDK-A1-Uniform').absolute(), path_to_urdf, 
                'package://a1_description/', '', 'mlp', 150, normalize, path_to_urdf_dynamics)
    unseen_all_dataset = torch.utils.data.ConcatDataset([torch.utils.data.Subset(uniform, np.arange(0, uniform.__len__() - 1))])
    pred_mlp, labels_mlp, mse, rmse, l1 = evaluate_model(mlp_path, torch.utils.data.Subset(unseen_all_dataset, np.arange(0, unseen_all_dataset.__len__())))

    # Evaluate Unseen All on Dynamics
    uniform = QuadSDKDataset_A1_Uniform(Path(Path('.').parent, 'datasets', 'QuadSDK-A1-Uniform').absolute(), path_to_urdf, 
                'package://a1_description/', '', 'dynamics', 1, normalize, path_to_urdf_dynamics)
    unseen_all_dataset = torch.utils.data.ConcatDataset([torch.utils.data.Subset(uniform, np.arange(148, uniform.__len__()))])
    pred_dynamics, labels_dynamics, mse, rmse, l1 = evaluate_model(None, torch.utils.data.Subset(unseen_all_dataset, np.arange(0, unseen_all_dataset.__len__())))

    # Get the time labels
    s_i = 12250
    s_e = s_i + 600
    uniform = QuadSDKDataset_A1_Uniform(Path(Path('.').parent, 'datasets', 'QuadSDK-A1-Uniform').absolute(), path_to_urdf, 
                'package://a1_description/', '', 'mlp', 150, normalize, path_to_urdf_dynamics)
    x_times = []
    for i in range(s_i, s_e):
        lin_acc, ang_vel, j_p, j_v, j_T, f_p, f_v, labels, r_p, r_o, ts = uniform.load_data_sorted(i)

        # Take average of timestamps to assign this entry a timestamp value
        ts_avg = np.average(ts[-1])
        x_times.append(ts_avg)

    # Shift so it starts at 0
    x_times = x_times - x_times[0]

    # Setup four graphs (one for each foot)
    fig, axes = plt.subplots(4, figsize=[10, 5])

    # Display the results
    plt.rcParams.update({
        "text.usetex": True
    })
    titles = [
        'GRF_FL (N)', 
        'GRF_RL (N)', 
        'GRF_FR (N)', 
        'GRF_RR (N)'
    ]
    colors = sns.color_palette("tab10")
    for i in range(0, 4):
        axes[i].plot(x_times, labels_mlp[s_i:s_e, i], label="Ground Truth", color=colors[3], linestyle='-.')
        axes[i].plot(x_times, pred_dynamics[s_i:s_e, i], label="Floating Base Dynamics", color=colors[0])
        axes[i].plot(x_times, pred_mlp[s_i:s_e, i], label="MLP", color=colors[2])
        axes[i].plot(x_times, pred_hgnn[s_i:s_e, i], label="MI-HGNN", color=colors[1])
        if i != 3:
            axes[i].get_xaxis().set_visible(False)
        if i == 0:
            axes[i].legend()
        axes[i].set_ylabel(titles[i], fontsize=10)
        if i == 3:
            axes[i].set_xlabel("Time (s)", fontsize=10)

    # Save the figure
    plt.savefig("paper/regression_evaluation.png", bbox_inches='tight', pad_inches=0.01, dpi=200)

def main():
    evaluation_example()

if __name__ == "__main__":
    main()