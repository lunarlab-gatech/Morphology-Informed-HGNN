import torch
from torch import nn
from torch_geometric.nn import Linear, HeteroConv, HeteroDictLinear, GraphConv
import numpy as np

class GRF_HGNN(torch.nn.Module):
    """
    The Ground Reaction Force Heterogeneous Graph Neural Network model.
    """

    def __init__(self, hidden_channels: int, num_layers: int, data_metadata, 
                 regression: bool = True, activation_fn = nn.ReLU()):
        """
        Implementation of the MI-HGNN model. 

        Parameters:
            hidden_channels (int): Size of the node embeddings in the graph.
            num_layers (int): Number of message-passing layers.
            data_metadata (tuple): Contains information on the node and edge types
                in the graph, which is returned by get_data_metadata() from dataset
                class.
            regression (bool): True if the problem is regression, false if 
                classification. This changes the output values of the model, whether 
                they are GRF values or contact probability logits, respectively.
            activation_fn (class): The activation function used between the layers.
        """

        super().__init__()
        self.regression = regression
        self.activation = activation_fn

        # Create the first layer encoder to convert features into embeddings
        # Just does node features
        self.encoder = HeteroDictLinear(-1, hidden_channels, data_metadata[0])

        # Create a convolution for each layer
        self.convs = torch.nn.ModuleList()
        for _ in range(num_layers):
            conv_dict = {}
            for edge_connection in data_metadata[1]:
                conv_dict[edge_connection] = GraphConv(hidden_channels,
                                                    hidden_channels,
                                                    aggr='add')
            conv = HeteroConv(conv_dict, aggr='sum')
            self.convs.append(conv)

        # Create the final linear layer (Decoder) -> Just for nodes of type "foot"
        if self.regression: # For GRF values
            self.out_channels_per_foot = 1
        else: # For contact probability logits
             self.out_channels_per_foot = 2
        self.decoder = Linear(hidden_channels, self.out_channels_per_foot)

    def forward(self, x_dict, edge_index_dict):
        x_dict = self.encoder(x_dict)
        x_dict = {key: self.activation(x) for key, x in x_dict.items()}
        for conv in self.convs:
            x_dict = conv(x_dict, edge_index_dict)
            x_dict = {key: self.activation(x) for key, x in x_dict.items()}
        return self.decoder(x_dict['foot'])