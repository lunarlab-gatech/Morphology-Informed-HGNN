import torch
from torch import nn
from torch_geometric.nn import Linear, HeteroConv, HeteroDictLinear, GraphConv
import numpy as np

class GRF_HGNN(torch.nn.Module):
    """
    The Ground Reaction Force Heterogeneous Graph Neural Network model.
    """

    def __init__(self, hidden_channels: int, edge_dim: int, num_layers: int, 
                 out_channels: int, data_metadata, regression: bool = True, 
                 activation_fn = nn.ReLU()):
        """
        Parameters:
            out_channels (int): Only used for regression problems. The number
                of predicted output values.
            out_classes (int): Only used for classification problems. The number
                of predicted output classes PER FOOT. Ex: if out_classes is 2, 
                then the total # of output classes for all four legs is 2^4, or 16.
        """


        super().__init__()
        self.regression = regression

        # Create the first layer encoder to convert features into embeddings
        # Just does node features
        self.encoder = HeteroDictLinear(-1, hidden_channels, data_metadata[0])

        # Create a convolution for each layer
        self.convs = torch.nn.ModuleList()
        for _ in range(num_layers):
            conv_dict = {}
            for edge_connection in data_metadata[1]:
                # TODO: Maybe wap this out with an operation that better matches what we want to happen with the edges,
                # instead of just using it for the attention calculation.
                conv_dict[edge_connection] = GraphConv(hidden_channels,
                                                    hidden_channels,
                                                    aggr='add')
            conv = HeteroConv(conv_dict, aggr='sum')
            self.convs.append(conv)

        # Save the activation function
        self.activation = activation_fn

        # Create the final linear layer (Decoder) -> Just for nodes of type "foot"
        # Meant to calculate the final GRF values
        if self.regression:
            self.decoder = Linear(hidden_channels, out_channels)
        else: # Add a Softmax for classification problems
            modules = []
            modules.append(nn.Linear(hidden_channels, out_channels))
            modules.append(nn.Sigmoid())
            self.decoder = nn.Sequential(*modules)

    def forward(self, x_dict, edge_index_dict):
        x_dict = self.encoder(x_dict)
        x_dict = {key: self.activation(x) for key, x in x_dict.items()}
        for conv in self.convs:
            # TODO: Does the Activation function actually work?
            x_dict = conv(x_dict, edge_index_dict)
            x_dict = {key: self.activation(x) for key, x in x_dict.items()}
        return self.decoder(x_dict['foot'])