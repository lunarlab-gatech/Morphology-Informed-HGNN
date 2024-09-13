import pandas
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from pathlib import Path
from mi_hgnn.lightning_py.gnnLightning import train_model, evaluate_model
import torch
from mi_hgnn.datasets_py.quadSDKDataset import *
from mi_hgnn.visualization import visualize_model_outputs_regression
import pandas

def regression_bar_plot():
    # Read in all results
    df_mlps = pandas.read_csv("paper/regression_results_mlps.csv")
    df_hgnns = pandas.read_csv("paper/regression_results_hgnns.csv")
    df_dynamics = pandas.read_csv("paper/regression_results_dynamics.csv")

    # Add model type onto all results
    df_mlps.insert(1, "Type", ["MLP", "MLP", "MLP", "MLP", "MLP", "MLP", "MLP", "MLP"])
    df_hgnns.insert(1, "Type", ["MI-HGNN", "MI-HGNN", "MI-HGNN", "MI-HGNN", "MI-HGNN", "MI-HGNN", "MI-HGNN", "MI-HGNN"])
    df_dynamics.insert(1, "Type", ["Dynamics Model"])
    df = pandas.concat([df_mlps, df_hgnns, df_dynamics], ignore_index=True)

    # Pull out only RMSE losses into long-format
    type_list, metric_list, score_list = [], [], []
    metrics_to_extract = ['F-RMSE', 'S-RMSE', 'T-RMSE', 'A-RMSE', 'Full-RMSE']
    metrics_to_extract_readable = ['Unseen\nFriction', 'Unseen\nSpeed', 'Unseen\nTerrain', 'Unseen\nAll', 'Unseen\nAverage']
    for i in range(0, df.shape[0]):
        row = df.iloc[i]
        for i, metric in enumerate(metrics_to_extract):
            type_list.append(row['Type'])
            metric_list.append(metrics_to_extract_readable[i])
            score_list.append(row[metric])
    dict = {"Type": type_list, "Metric": metric_list ,"RMSE": score_list}
    df_long = pandas.DataFrame(dict)

    # Plot the results
    H, W = 9, 10
    def cm2inch(cm):
        return cm / 2.54
    model_types = ["MI-HGNN", "MLP", "Dynamics Model"]
    palette = sns.color_palette("bright")
    palette =[palette[8], palette[2], palette[0]]
    fig, ax = plt.subplots(figsize=(cm2inch(W) * 1.25, cm2inch(H)), dpi=210)
    sns.barplot(data=df_long, x='Metric', y='RMSE',
                       hue='Type', hue_order=model_types,
                       errwidth=1.5, linewidth=0, capsize=.025,
                       ax=ax, ci="sd",
                       palette=palette)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.yaxis.grid(visible=True, alpha=0.1)
    ax.set_yscale('log')
    #ax.set_ylim([5, 50])
    ax.set_xlabel('')
    ax.tick_params(axis='both', which='major', labelsize=9)
    plt.setp(ax.get_legend().get_texts(), fontsize='9')
    plt.tight_layout()
    plt.savefig('paper/regression_plot.png')

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
    s_i = 10250
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
    fig, axes = plt.subplots(2, 2, figsize=[15, 7.5])

    # Display the results
    titles = [
        "FL Forces", "RL Forces", "FR Forces", "RR Forces"
    ]
    colors = sns.color_palette("bright")
    for i in range(0, 4):
        axes[int(i / 2), i % 2].plot(x_times, labels_mlp[s_i:s_e, i], label="Ground Truth", color=colors[3], linestyle='-.')
        axes[int(i / 2), i % 2].plot(x_times, pred_dynamics[s_i:s_e, i], label="Dynamics Model", color=colors[0])
        axes[int(i / 2), i % 2].plot(x_times, pred_mlp[s_i:s_e, i], label="MLP", color=colors[2])
        axes[int(i / 2), i % 2].plot(x_times, pred_hgnn[s_i:s_e, i], label="MI-HGNN", color=colors[8])
        # axes[int(i / 2), i % 2].set_xticklabels([])
        if i == 0:
            axes[int(i / 2), i % 2].legend()
        axes[int(i / 2), i % 2].set_title(titles[i])
        if i == 0 or i == 2:
            axes[int(i / 2), i % 2].set_ylabel("Ground Reaction Force (Z)")
        if i == 2 or i == 3:
            axes[int(i / 2), i % 2].set_xlabel("Time (s)")

    # Save the figure
    plt.savefig("paper/regression_evaluation.png", bbox_inches='tight', pad_inches=0.01, dpi=200)

def main():
    regression_bar_plot()
    evaluation_example()

if __name__ == "__main__":
    main()