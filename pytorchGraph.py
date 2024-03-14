import torch
import torch_geometric
from torch_geometric.data import Data
from urdfParser import RobotURDF
import networkx
import matplotlib.pyplot as plt

def main():
    # Load the HyQ URDF file
    HyQ_URDF = RobotURDF('urdf_files/HyQ/hyq.urdf') 

    # Put Joint information into Nodes #TODO

    # Extract the edge matrix and convert to tensor
    edge_matrix = torch.tensor(HyQ_URDF.get_edge_index(), dtype=torch.long)

    # Create the graph
    graph = Data(edge_index=edge_matrix, num_nodes=HyQ_URDF.get_num_nodes())

    # Convert to networkx graph
    nx_graph = torch_geometric.utils.to_networkx(graph, to_undirected=True)

    # Draw the graph
    spring_layout = networkx.spring_layout(nx_graph)
    networkx.draw(nx_graph, pos=spring_layout)
    networkx.draw_networkx_labels(nx_graph, pos=spring_layout, labels=HyQ_URDF.get_node_name_dict(), verticalalignment='top',
                                  font_size=8)
    networkx.draw_networkx_edge_labels(nx_graph, pos=spring_layout, edge_labels=HyQ_URDF.get_edge_name_dict(), rotate=False,
                                       font_size=7)
    plt.show()

if __name__ == "__main__":
    main()
