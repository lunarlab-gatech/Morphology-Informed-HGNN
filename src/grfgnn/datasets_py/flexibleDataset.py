import torch
from torch_geometric.data import Data, Dataset, HeteroData
from ..graphParser import NormalRobotGraph, HeterogeneousRobotGraph
from rosbags.highlevel import AnyReader
from pathlib import Path
from torchvision.datasets.utils import download_file_from_google_drive
import os

class FlexibleDataset(Dataset):
    """
    Dataset class that can be used for a MLP, normal GNN, or 
    heterogeneous GNN.
    """

    # Used when initalizing the parent class instead of a proper
    # child classs
    notImplementedError = NotImplementedError(
        "Don't call this class directly, but use one of the child classes in order to choose which dataset sequence you want to load."
    )

    def __init__(self,
                 root: Path,
                 urdf_path: Path,
                 ros_builtin_path: str,
                 urdf_to_desc_path: str,
                 data_format: str = 'gnn',
                 history_length: int = 1,
                 transform=None,
                 pre_transform=None,
                 pre_filter=None):
        """
        Parameters:
                history_length (int): The length of the history of inputs to use
                    for our graph entries.
                data_format (str): Either 'gnn', 'mlp', or 'heterogeneous_gnn'. 
                    Determines how the get() method returns data.
        """
        # Check for valid data format
        self.data_format = data_format
        if self.data_format != 'mlp' and self.data_format != 'gnn' and self.data_format != 'heterogeneous_gnn':
            raise ValueError(
                "Parameter 'data_format' must be 'gnn', 'mlp', or 'heterogeneous_gnn'."
            )

        # Setup the directories for raw and processed data
        self.root = root
        super().__init__(str(root), transform, pre_transform, pre_filter)

        # Get the first index and length of dataset
        info_path = Path(root, 'processed', 'info.txt')
        with open(info_path, 'r') as f:
            data = f.readline().split(" ")

            # The longer the history, technically the lower number of total dataset entries
            # we can use, and thus a lower dataset length.
            self.history_length = history_length
            self.length = int(data[0]) - (self.history_length - 1)
            if self.length <= 0:
                raise ValueError(
                    "Dataset has too few entries for the provided 'history_length'."
                )
            self.first_index = int(data[1])

        # Parse the robot graph from the URDF file
        if self.data_format == 'heterogeneous_gnn':
            self.robotGraph = HeterogeneousRobotGraph(urdf_path,
                                                      ros_builtin_path,
                                                      urdf_to_desc_path)
        else:
            self.robotGraph = NormalRobotGraph(urdf_path, ros_builtin_path,
                                               urdf_to_desc_path)

        # Make sure the URDF file we were given properly matches
        # what we are expecting
        passed_urdf_name = self.robotGraph.robot_urdf.name
        expected_name = self.get_expected_urdf_name()
        if passed_urdf_name != expected_name:
            raise ValueError(
                "Invalid URDF: \"" + passed_urdf_name +
                "\". This dataset was collected on the \"" + expected_name +
                "\" robot. Please pass in the URDF file for the \"" +
                expected_name + "\" robot, not for the \"" + passed_urdf_name +
                "\" robot.")

        # Get node name to index mapping
        self.urdf_name_to_graph_index = self.robotGraph.get_node_name_to_index_dict(
        )

    def get_data_format(self) -> str:
        """
        Returns the data format of the get() method.
        """
        return self.data_format

    def get_google_drive_file_id(self):
        """
        Method for child classes to choose which sequence to load;
        used if the dataset is
        """
        raise self.notImplementedError

    @property
    def raw_file_names(self):
        return ['data.bag']

    def download(self):
        download_file_from_google_drive(self.get_google_drive_file_id(),
                                        Path(self.root, 'raw'), "data.bag")

    @property
    def processed_file_names(self):
        """
        Make the list of all the processed 
        files we are expecting.
        """

        processed_file_names = []
        start_seq_id, end_seq_id = self.get_start_and_end_seq_ids()
        for i in range(start_seq_id, end_seq_id + 1):
            processed_file_names.append(str(i) + ".txt")
        processed_file_names.append("info.txt")
        return processed_file_names

    def process(self):
        raise self.notImplementedError

    def get_foot_node_indices_matching_labels(self):
        """
        This helper method returns the foot node indices that correspond 
        to the ground truth GRF labels contained in the "y" of each 
        Data object. The number x in the ith index tells us that 
        ground truth label i matches up with node x from the GNN output.

        For example, if the graph has 8 nodes, and there are only 4 
        ground truth labels, this method might return [0, 3, 4, 7].
        This means that the first ground truth label should match
        the output of the node at index 0, the second ground truth label
        should match the output of the node at index 3, and so on.

        Returns:
            foot_node_indices (list[int]) - List of the indices of
            the graph nodes that should output the ground truth labels 
            at the same corresponding index.
        """
        if self.foot_node_indices is None:
            raise self.notImplementedError
        else:
            return self.foot_node_indices

    def get_start_and_end_seq_ids(self):
        """
        Method for child classes to tell the processing method the
        numbers of the files they are looking for. It returns
        the seq id of the first ROS '/hardware_a1/joint_foot' 
        message, and the seq id of the last.
        """
        raise self.notImplementedError

    def get_expected_urdf_name(self):
        """
        Method for child classes to specify what URDF file they are
        looking for.
        """
        raise self.notImplementedError

    def len(self):
        return self.length

    def get_helper_mlp(self, idx):
        """
        Gets a Dataset entry if we are using an MLP model.
        """
        raise self.notImplementedError

    def get_helper_gnn(self, idx):
        """
        Get a dataset entry if we are using a GNN model.
        """
        raise self.notImplementedError

    def get_helper_heterogeneous_gnn(self, idx):
        """
        Get a dataset entry if we are using a Heterogeneous GNN model.
        """
        raise self.notImplementedError

    def get(self, idx):
        # Bounds check
        if idx < 0 or idx >= self.length:
            raise IndexError("Index value out of Dataset bounds.")

        # Return data in the proper format
        if self.data_format == 'gnn':
            return self.get_helper_gnn(idx)
        elif self.data_format == 'mlp':
            return self.get_helper_mlp(idx)
        elif self.data_format == 'heterogeneous_gnn':
            return self.get_helper_heterogeneous_gnn(idx)