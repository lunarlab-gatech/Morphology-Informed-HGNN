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
import pinocchio as pin

def evaluation_video():

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
    unseen_all_dataset_hgnn = torch.utils.data.ConcatDataset([torch.utils.data.Subset(uniform, np.arange(0, uniform.__len__() - 1))])
    pred_hgnn, labels_hgnn, mse, rmse, l1 = evaluate_model(hgnn_path, torch.utils.data.Subset(unseen_all_dataset_hgnn, np.arange(0, unseen_all_dataset_hgnn.__len__())))

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
    uniform_time = QuadSDKDataset_A1_Uniform(Path(Path('.').parent, 'datasets', 'QuadSDK-A1-Uniform').absolute(), path_to_urdf, 
                'package://a1_description/', '', 'heterogeneous_gnn', 32311, normalize, path_to_urdf_dynamics)
    unseen_time = torch.utils.data.ConcatDataset([torch.utils.data.Subset(uniform_time, np.arange(0, uniform_time.__len__() - 1))])
    x_times = []

    lin_acc, ang_vel, j_p, j_v, j_T, f_p, f_v, labels, r_p, r_o, ts = unseen_time.datasets[0].dataset.load_data_sorted(0)

    # Take average of timestamps to assign this entry a timestamp value
    ts_avg = np.average(ts, axis=1)
    ts_avg = ts_avg[149:]
    x_times = ts_avg
    np.testing.assert_equal(x_times.shape[0], 32162)
        
    # Now xtimes has the exactly timestamps used in the dataset evaluation, consistent across all models

    # Shift so it starts at 0
    x_times = x_times - x_times[0]

    # Setup four graphs (one for each foot)
    fig, axes = plt.subplots(1, figsize=[9, 4])
    titles = [
        'GRF_FL (N)', 
        'GRF_RL (N)', 
        'GRF_FR (N)', 
        'GRF_RR (N)'
    ]
    colors = sns.color_palette("tab10")
    line = []
    axes = [axes]
    i = 0
    for i in range(0, 1):
        line.append(axes[i].plot([], [], label="Ground Truth", color=colors[3], lw=2, linestyle='-.')[0])
        line.append(axes[i].plot([], [], label="Floating Base Dynamics", color=colors[0])[0])
        line.append(axes[i].plot([], [], label="MLP", color=colors[2])[0])
        line.append(axes[i].plot([], [], label="MI-HGNN", color=colors[1])[0])
        axes[i].set_ylim(-1, 150)
        axes[i].set_ylabel(titles[i], fontsize=10)
        axes[i].set_xlabel("Time (s)", fontsize=10)
    line.append(axes[i].axvline(x=0, color='black', label='avxline - full height'))

    # Setup the pinocchio model
    dataset_raw: FlexibleDataset = unseen_all_dataset.datasets[0].dataset
    urdf_path = Path(dataset_raw.robotGraphFull.new_urdf_path)
    model, collision_model, visual_model = pin.buildModelsFromUrdf(
        str(urdf_path), str(urdf_path.parent.parent), pin.JointModelFreeFlyer(), verbose=True
    )
    data = model.createData()

    # Setup a viewer
    viz = pin.visualize.MeshcatVisualizer(model, collision_model, visual_model)
    viz.initViewer(loadModel=True)
    import time
    time.sleep(5)

    # Display the results
    from matplotlib.animation import FuncAnimation
    def update(frame):
        # Don't plot too much at once
        s = 0
        if frame - s > 250:
            s = frame - 250

        # Update y limits
        for axe in axes:
            ymin, ymax = axe.get_ylim()
            if frame > 0:
                amax = np.max(pred_dynamics[s:frame, 0].tolist())
                if amax > ymax:
                    axe.set_ylim(-1, amax)
        
        # Update x limits
        for axe in axes:
            xmin, xmax = axe.get_xlim()
            axe.set_xlim(x_times[s], x_times[frame - 1])

        # Update the pinocchio model (125 frames behind)
        pin_frame = frame - 125
        if pin_frame < 0: pin_frame = 0
        q, vel, acc, tau, y = unseen_all_dataset.__getitem__(pin_frame)
        q = q.cpu().numpy()
        pin.framesForwardKinematics(model, data, q)
        viz.display(q)

        # Draw a line at the point where pinocchio just updated
        line[4].set_xdata(x_times[pin_frame])

        # Update data
        for i in range(0, 1):
            line[4*i+0].set_data(x_times[s:frame], labels_mlp[s:frame, i].tolist())
            line[4*i+1].set_data(x_times[s:frame], pred_dynamics[s:frame, i])
            line[4*i+2].set_data(x_times[s:frame], pred_mlp[s:frame, i])
            line[4*i+3].set_data(x_times[s:frame], pred_hgnn[s:frame, i])
            axe.figure.canvas.draw()
        return line

    ani = FuncAnimation(fig, update, frames=len(x_times)+1, blit=True, repeat=False, interval=1)
    plt.show()

def main():
    evaluation_video()

if __name__ == "__main__":
    main()