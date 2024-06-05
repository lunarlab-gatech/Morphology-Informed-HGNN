from .datasets import QuadSDKDataset
from .graphParser import NormalRobotGraph
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch_geometric
from torch_geometric.data import Data, HeteroData
import networkx

def visualize_derivatives(dataset: QuadSDKDataset, num_to_visualize=1000):
    """
    This helper method visualizes the derivatives, to make sure that they aren't noisy.
    """

    # Extract joint acceleration data
    dataset_len = dataset.len()
    joint_vel_by_joint = [[], [], [], [], [], [], [], [], [], [], [], []]
    joint_acc_by_joint = [[], [], [], [], [], [], [], [], [], [], [], []]
    for i in range(0, dataset_len):
        graph: HeteroData = dataset.get(i)
        joint_attr: torch.Tensor = graph['joint'].x
        joint_acc = joint_attr[:,2]
        joint_vel = joint_attr[:,1]
        for j in range(0, 12):
            joint_acc_by_joint[j].append(joint_acc[j])
            joint_vel_by_joint[j].append(joint_vel[j])
        
    # Setup 12 graphs
    fig, axes = plt.subplots(12, figsize=[15, 40])
    # fig.suptitle('Calculated Joint Accelerations via Derivative')

    # Get the titles
    index_to_name_dict = dataset.robotGraph.get_node_index_to_name_dict('joint')
    titles = []
    for i in range(0, 12):
        titles.append(index_to_name_dict[i])

    # Plot the results
    for i in range(0, 12):
        color = 'tab:orange'
        axes[i].plot(np.array(joint_acc_by_joint[i], dtype=np.float64)[0:num_to_visualize], linestyle='-.', color=color)
        axes[i].set_title(titles[i])
        axes[i].set_xlabel("Dataset Entry  (about 530 per second)")
        axes[i].set_ylabel("Joint Acceleration (Calculated)", color=color)
        axes[i].tick_params(axis='y', labelcolor=color)

        ax_twin = axes[i].twinx()
        color = 'tab:blue'
        ax_twin.plot(np.array(joint_vel_by_joint[i], dtype=np.float64)[0:num_to_visualize], color=color)
        ax_twin.set_ylabel("Joint Velocity", color=color)
        ax_twin.tick_params(axis='y', labelcolor=color)

    plt.tight_layout()
    plt.savefig("DerivativesFig.pdf")

def visualize_graph(pytorch_graph: Data,
                    robot_graph: NormalRobotGraph,
                    fig_save_path: Path = None,
                    draw_edges: bool = False):
    """
    This helper method visualizes a Data graph object using networkx.
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
