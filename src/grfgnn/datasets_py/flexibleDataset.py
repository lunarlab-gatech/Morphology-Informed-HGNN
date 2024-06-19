import torch
from torch_geometric.data import Data, Dataset, HeteroData
from ..graphParser import NormalRobotGraph, HeterogeneousRobotGraph
from rosbags.highlevel import AnyReader
from pathlib import Path
from torchvision.datasets.utils import download_file_from_google_drive
import os
import numpy as np

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
                data_format (str): Either 'gnn', 'mlp', or 'heterogeneous_gnn'. 
                    Determines how the get() method returns data.
                history_length (int): The length of the history of inputs to use
                    for our graph entries.
        """
        # Check for valid data format
        self.data_format = data_format
        if self.data_format != 'mlp' and self.data_format != 'gnn' and self.data_format != 'heterogeneous_gnn':
            raise ValueError(
                "Parameter 'data_format' must be 'gnn', 'mlp', or 'heterogeneous_gnn'."
            )

        # Setup the directories for raw and processed data, download it, 
        # and process
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
        self.urdf_name_to_graph_index = self.robotGraph.get_node_name_to_index_dict()

        # Define the order that the sorted joint and foot data should be in
        self.foot_node_sorted_order = self.get_foot_node_sorted_order()

        # Precompute the array indexes matching the defined orders for joints and feet
        self.joint_node_indices_sorted = []
        for urdf_node_name in self.get_joint_node_sorted_order():
            self.joint_node_indices_sorted.append(self.urdf_name_to_dataset_array_index[urdf_node_name])
        self.foot_node_indices_sorted = []
        for urdf_node_name in self.get_foot_node_sorted_order():
            self.foot_node_indices_sorted.append(self.urdf_name_to_dataset_array_index[urdf_node_name])

        # Precompute the return value for get_foot_node_indices_matching_labels()
        self.foot_node_indices_of_graph_matching_labels = []
        for urdf_name in self.get_foot_node_sorted_order():
            self.foot_node_indices_of_graph_matching_labels.append(self.urdf_name_to_graph_index[urdf_name])

    def get_urdf_name_to_dataset_array_index(self) -> dict:
        """
        Returns a dictionary that maps strings of
        URDF joint names to the corresponding index
        in the dataset array. This allows us to know
        which dataset entries correspond to the which 
        joints in the URDF.
        """
        raise self.notImplementedError

    def get_joint_node_sorted_order(self) -> list[str]:
        """
        Returns an array with the names of the URDF joints
        corresponding to actual joints on the robot.

        They are arranged in the order we want the data sorted
        for the load_data_sorted() method.
        """
        raise self.notImplementedError

    def get_foot_node_sorted_order(self) -> list[str]:
        """
        Returns an array with the names of the URDF
        joints corresponding to the robot feet.
        
        They are arranged in the order we want the data sorted
        for the load_data_sorted() method.
        """
        raise self.notImplementedError

    def get_data_format(self) -> str:
        """
        Returns the data format of the get() method.
        """
        return self.data_format

    def get_google_drive_file_id(self):
        """
        Method for child classes to choose which sequence to load;
        used if the dataset is downloaded.
        """
        raise self.notImplementedError
    
    def get_downloaded_dataset_file_name(self):
        """
        Method for defining the new name of the downloaded
        dataset file.
        """
        raise self.notImplementedError

    @property
    def raw_file_names(self):
        return [self.get_downloaded_dataset_file_name()]

    def download(self):
        download_file_from_google_drive(self.get_google_drive_file_id(),
                                        Path(self.root, 'raw'),
                                        self.get_downloaded_dataset_file_name())

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
        return self.foot_node_indices_of_graph_matching_labels


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
    
    def load_data_at_dataset_seq(self, seq_num: int):
        """
        This helper function opens the txt file at "ros_seq"
        and loads dataset information for that sequence.

        Parameters:
            seq_num (int): The sequence number of the txt file
                whose data should be loaded.

        Returns:
            lin_acc - IMU linear acceleration
            ang_vel - IMU angular velocity
            j_p - Joint positions 
            j_v - Joint velocities
            j_T - Joint Torques
            f_p - Foot position
            f_v - Foot velocity
            labels - The Dataset labels (either Z direction GRFs, or contact states) 

            NOTE: If the dataset doesn't have a certain value, or we aren't currently
            using it, the parameter will be filled with a value of None.
        """
        # TODO: Make sure child classes return these as NUMPY arrays
        
        raise self.notImplementedError
    
    def load_data_sorted(self, seq_num: int):
        """
        Loads data from the dataset at the provided sequence number.
        However, the joint and feet information are sorted so that 
        they match the order returned by get_joint_node_sorted_order().

        Additionally, the labels are sorted so it matches the order
        returned by get_foot_node_sorted_order().

        Parameters:
            seq_num (int): The sequence number of the txt file
                whose data should be loaded.

        Returns:
            Same values as load_data_at_dataset_seq(), but order of
            values inside arrays have been sorted.
        """
        lin_acc, ang_vel, j_p, j_v, j_T, f_p, f_v, labels = self.load_data_at_dataset_seq(seq_num)

        # Sort the joint information
        unsorted_list = [j_p, j_v, j_T, f_p, f_v]
        sorted_list = []
        for unsorted_array in unsorted_list:
            if unsorted_array is not None:
                sorted_list.append(unsorted_array[self.joint_node_indices_sorted])
            else:
                sorted_list.append(None)

        # Sort the ground truth labels
        labels_sorted = labels[self.foot_node_indices_sorted]

        return lin_acc, ang_vel, sorted_list[0], sorted_list[1], sorted_list[2], sorted_list[3], sorted_list[4], labels_sorted

    def get_helper_mlp(self, idx: int):
        """
        Gets a Dataset entry if we are using an MLP model.
        """

        # Make the network inputs
        x = None

        # Find out which variables we have to use
        lin_acc, ang_vel, j_p, j_v, j_T, f_p, f_v, labels = self.load_data_sorted(self.first_index + idx + i)
        variables_to_check = [lin_acc, ang_vel, j_p, j_v, j_T, f_p, f_v]
        variables_to_use = []
        for i, variable in enumerate(variables_to_check):
            if variable is not None:
                variables_to_use.append(i)
        variables_to_use = np.array(variables_to_use)

        # Load the dataset information information
        for i in range(0, self.history_length):

            # Only add variables that aren't None
            lin_acc, ang_vel, j_p, j_v, j_T, f_p, f_v, labels = self.load_data_sorted(self.first_index + idx + i)
            dataset_inputs = [lin_acc, ang_vel, j_p, j_v, j_T, f_p, f_v]
            final_input = []
            for y in variables_to_use:
                final_input = final_input + dataset_inputs[y] 

            # Construct the input tensor
            tensor = torch.tensor((final_input), 
                                  dtype=torch.float64).unsqueeze(0)
            if x is None: x = tensor
            else: x = torch.cat((x, tensor), 0)

        # Flatten the tensor if necessary
        if len(x.size()) > 1:
            x = torch.flatten(torch.transpose(x, 0, 1), 0, 1)

        # Create the ground truth labels
        lin_acc, ang_vel, j_p, j_v, j_T, f_p, f_v, labels = self.load_data_sorted(self.first_index + idx + self.history_length - 1)
        y = torch.tensor(labels, dtype=torch.float64)
        return x, y

    def get_helper_gnn(self, idx: int):
        """
        Get a dataset entry if we are using a GNN model.
        """

        # Create a note feature matrix
        x = torch.ones((self.robotGraph.get_num_nodes(), 4 * self.history_length), dtype=torch.float64)

        # For each dataset entry we include in the history
        for j in range(0, self.history_length):
            # Load the data for this entry
            lin_acc, ang_vel, positions, velocities, torques, z_grfs, ang_acc, joint_acc = self.load_data_sorted(self.first_index + idx + j)

            # For each joint specified
            for i, urdf_node_name in enumerate(self.joint_nodes_for_attributes):

                # Find the index of this particular node
                node_index = self.urdf_name_to_graph_index[urdf_node_name]

                # Add the features to x matrix
                x[node_index, 0*self.history_length+j] = positions[i]
                x[node_index, 1*self.history_length+j] = velocities[i]
                x[node_index, 2*self.history_length+j] = joint_acc[i]
                x[node_index, 3*self.history_length+j] = torques[i]

        # Create the edge matrix
        self.edge_matrix = self.robotGraph.get_edge_index_matrix()
        self.edge_matrix_tensor = torch.tensor(self.edge_matrix,
                                               dtype=torch.long)

        # Create the labels
        lin_acc, ang_vel, positions, velocities, torques, z_grfs, ang_acc, joint_acc = self.load_data_sorted(
            self.first_index + idx + self.history_length - 1)
        y = torch.tensor(z_grfs, dtype=torch.float64)

        # Create the graph
        graph = Data(x=x, edge_index=self.edge_matrix_tensor,
                     y=y, num_nodes=self.robotGraph.get_num_nodes())
        return graph

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