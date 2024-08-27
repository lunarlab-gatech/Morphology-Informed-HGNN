from .datasets_py.quadSDKDataset import QuadSDKDataset
from .graphParser import NormalRobotGraph
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch_geometric
from torch_geometric.data import Data, HeteroData
import networkx
from torchmetrics import ConfusionMatrix

def display_on_axes(axes, estimated, ground_truth, title):
    """
    Simple function that displays ground truth and estimated
    information on a Matplotlib.pyplot Axes.
    """
    axes.plot(ground_truth, label="Ground Truth", linestyle='-.')
    axes.plot(estimated, label="Predicted")
    axes.legend()
    axes.set_title(title)

def visualize_model_outputs_regression(pred, labels, path_to_file: Path = None):
    """
    Helper method that creates a figure between the predicted
    GRF values and the actual GRF values, and saves it at 
    'path_to_file'.
    """

    # Setup four graphs (one for each foot)
    fig, axes = plt.subplots(4, figsize=[20, 10])
    fig.suptitle('Foot Predicted GRF Forces vs. Ground Truth')

    # Display the results
    titles = [
        "Foot 0 Forces", "Foot 1 Forces",
        "Foot 2 Forces", "Foot 3 Forces"
    ]
    for i in range(0, 4):
        display_on_axes(axes[i], pred[:, i], labels[:, i], titles[i])

    # Save the figure
    if path_to_file is not None:
        plt.savefig(path_to_file)


def visualize_model_outputs_classification(y_pred_per_foot_prob_only_1: torch.Tensor, 
                                           labels: torch.Tensor, path_to_file: Path = None, 
                                           fig_width: int = 20):
    """
    Helper method that plots the difference between the predicted class
    and the actual.
    TODO: Plot the difference between 16 state contact and predicted.
    TODO: Generate a confusion matrix.

    Parameters:
        - y_pred_per_foot_prob_only_1 (torch.Tensor): A Tensor of shape 
            (batch_size, 4) that contains probability values of stable 
            contact for each foot.
        - labels (torch.Tensor): A Tensor of shape (batch_size, 4) that 
            contains contact state for each foot.
    """

    # Setup four graphs (one for each foot)
    fig, axes = plt.subplots(4, figsize=[fig_width, 20])
    fig.suptitle('Foot Predicted Contact States vs. Ground Truth')

    # Display the results
    titles = [
        "Foot 0 State", "Foot 1 State",
        "Foot 2 State", "Foot 3 State"
    ]
    for i in range(0, 4):
        display_on_axes(axes[i], np.rint(y_pred_per_foot_prob_only_1[:, i]), labels[:, i], titles[i])

    if path_to_file is not None:
        plt.savefig(path_to_file)

def visualize_dataset_graph(pytorch_graph: Data,
                    robot_graph: NormalRobotGraph,
                    fig_save_path: Path = None,
                    draw_edges: bool = False):
    """
    This helper method visualizes a Data graph object using networkx.
    Only works for Data objects created with the 'gnn' data_format.
    """

    # Write the features onto the names
    node_labels = robot_graph.get_node_index_to_name_dict()
    for i in range(0, len(pytorch_graph.x)):
        label = node_labels[i]
        label += ": " + str(pytorch_graph.x[i].numpy())
        node_labels[i] = label

    # Convert to networkx graph
    nx_graph = torch_geometric.utils.to_networkx(pytorch_graph,
                                                 to_undirected=True)

    # Draw the graph
    spring_layout = networkx.spring_layout(nx_graph)
    networkx.draw(nx_graph, pos=spring_layout)
    networkx.draw_networkx_labels(nx_graph,
                                  pos=spring_layout,
                                  labels=node_labels,
                                  verticalalignment='top',
                                  font_size=8)
    if draw_edges:
        networkx.draw_networkx_edge_labels(
            nx_graph,
            pos=spring_layout,
            edge_labels=robot_graph.get_edge_connections_to_name_dict(),
            rotate=False,
            font_size=7)

    # Save the figure if requested
    if fig_save_path is not None:
        plt.savefig(fig_save_path)
    plt.show()
