import enum
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
    Dataset class that takes in robotic datasets with proprioceptive
    input information and ground truth labels. Flexible, as it can be 
    used for MLP, normal GNN, or heterogeneous GNN training. Additionally,
    can support datasets with a variety of inputs. 
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
                root (Path): The path to where the dataset is found (or where
                    it should be saved).
                urdf_path (Path): The path to the URDF file corresponding to the
                    robot this data was collected on.
                ros_builtin_path (str): The path ROS uses in the urdf file to navigate
                    to the urdf description directory. An example looks like this:
                    "package://a1_description/". You can find this by manually looking
                    in the urdf file.
                urdf_to_desc_path (str): The relative path from the urdf file to
                    the urdf description directory. This directory typically contains
                    folders like "meshes" and "urdf".
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
            self.joint_node_indices_sorted.append(self.get_urdf_name_to_dataset_array_index()[urdf_node_name])
        self.foot_node_indices_sorted = []
        for urdf_node_name in self.get_foot_node_sorted_order():
            self.foot_node_indices_sorted.append(self.get_urdf_name_to_dataset_array_index()[urdf_node_name])

        # Precompute the return value for get_foot_node_indices_matching_labels()
        self.foot_node_indices_of_graph_matching_labels = []
        for urdf_name in self.get_foot_node_sorted_order():
            self.foot_node_indices_of_graph_matching_labels.append(self.urdf_name_to_graph_index[urdf_name])

    # ================================================================
    # ========================= DOWNLOADING ==========================
    # ================================================================

    @property
    def raw_file_names(self):
        return [self.get_downloaded_dataset_file_name()]

    def download(self):
        download_file_from_google_drive(self.get_google_drive_file_id(),
                                        Path(self.root, 'raw'),
                                        self.get_downloaded_dataset_file_name())
        
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
    
    # ================================================================
    # ========================= PROCESSING ===========================
    # ================================================================

    @property
    def processed_file_names(self):
        """
        Make the list of all the processed 
        files we are expecting.
        """
        return ["info.txt"]

    def process(self):
        """
        Called when we don't have all of the processed
        files we are expecting, so we process the data
        to generate those files.
        """
        raise self.notImplementedError
    
    # ================================================================
    # ============= DATA SORTING ORDER AND MAPPINGS ==================
    # ================================================================

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

    def get_base_node_sorted_order(self) -> list[str]:
        """
        Returns an array with the name of the URDF
        base corresponding to the robot center.
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
    
    def get_urdf_name_to_dataset_array_index(self) -> dict:
        """
        Returns a dictionary that maps strings of
        URDF joint names to the corresponding index
        in the dataset array. This allows us to know
        which dataset entries correspond to the which 
        joints in the URDF.
        """
        raise self.notImplementedError

    # ================================================================
    # ===================== DATASET PROPERTIES =======================
    # ================================================================

    def len(self):
        """
        Returns the number of dataset entries.
        Note that this changes depending on the
        self.history_length parameter.
        """
        return self.length
    
    def get_data_format(self) -> str:
        """
        Returns the data format of the get() method.
        """
        return self.data_format
    
    def get_data_metadata(self):
        """
        Returns the data metadata. Only for use with
        heterogeneous graph data.
        """
        if self.data_format != 'heterogeneous_gnn':
            raise TypeError(
                "This function is only for a data_format of 'heterogeneous_gnn'."
            )
        node_types = ['base', 'joint', 'foot']
        edge_types = [('base', 'connect', 'joint'),
                      ('joint', 'connect', 'base'),
                      ('joint', 'connect', 'joint'),
                      ('foot', 'connect', 'joint'),
                      ('joint', 'connect', 'foot')]
        return (node_types, edge_types)
    
    def get_expected_urdf_name(self):
        """
        Method for child classes to specify what URDF file they are
        looking for.
        """
        raise self.notImplementedError
    
    # ================================================================
    # ======================== DATA LOADING ==========================
    # ================================================================
    
    def load_data_sorted(self, seq_num: int):
        """
        Loads data from the dataset at the provided sequence number.
        However, the joint and feet information are sorted so that 
        they match the order returned by get_joint_node_sorted_order().

        Additionally, the labels are sorted so it matches the order
        returned by get_foot_node_sorted_order().

        Finally, labels are checked to make sure they aren't None.

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
        labels_sorted = None
        if labels is None:
            raise ValueError("Dataset must provide labels.")
        else:
            labels_sorted = labels[self.foot_node_indices_sorted]

        return lin_acc, ang_vel, sorted_list[0], sorted_list[1], sorted_list[2], sorted_list[3], sorted_list[4], labels_sorted

    def load_data_at_dataset_seq(self, seq_num: int):
        """
        This helper function opens the txt file at "ros_seq"
        and loads dataset information for that sequence.

        Parameters:
            seq_num (int): The sequence number of the txt file
                whose data should be loaded.

        Returns:
            lin_acc (np.array) - IMU linear acceleration
            ang_vel (np.array) - IMU angular velocity
            j_p (np.array) - Joint positions 
            j_v (np.array) - Joint velocities
            j_T (np.array) - Joint Torques
            f_p (np.array) - Foot position
            f_v (np.array) - Foot velocity
            labels (np.array) - The Dataset labels (either Z direction GRFs, or contact states) 

            NOTE: If the dataset doesn't have a certain value, or we aren't currently
            using it, the parameter will be filled with a value of None.
        """
        
        raise self.notImplementedError

    # ================================================================
    # =================== GET METHOD AND HELPERS =====================
    # ================================================================

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
        
    def get_helper_mlp(self, idx: int):
        """
        Gets a Dataset entry if we are using an MLP model.
        """

        # Make the network inputs
        x = None

        # Find out which variables we have to use
        lin_acc, ang_vel, j_p, j_v, j_T, f_p, f_v, labels = self.load_data_sorted(self.first_index + idx)
        variables_to_check = [lin_acc, ang_vel, j_p, j_v, j_T, f_p, f_v]
        variables_to_use = self.find_variables_to_use(variables_to_check)

        # Load the dataset information information
        for i in range(0, self.history_length):

            # Only add variables that aren't None
            lin_acc, ang_vel, j_p, j_v, j_T, f_p, f_v, labels = self.load_data_sorted(self.first_index + idx + i)
            dataset_inputs = [lin_acc, ang_vel, j_p, j_v, j_T, f_p, f_v]
            final_input = np.array([])
            for y in variables_to_use:
                final_input = np.concatenate((final_input, dataset_inputs[y]), axis=0) 

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

        # Find out which necessary variables are available
        lin_acc, ang_vel, j_p, j_v, j_T, f_p, f_v, labels = self.load_data_sorted(self.first_index + idx)
        variables_to_check = [j_p, j_v, j_T]
        variables_to_use = self.find_variables_to_use(variables_to_check)
        if len(variables_to_use) == 0:
            raise ValueError("Dataset must provide at least one input.")
        
        # Create a note feature matrix
        x = torch.ones((self.robotGraph.get_num_nodes(), len(variables_to_use) * self.history_length), dtype=torch.float64)

        # For each dataset entry we include in the history
        for j in range(0, self.history_length):
            # Load the data for this entry
            lin_acc, ang_vel, j_p, j_v, j_T, f_p, f_v, labels = self.load_data_sorted(self.first_index + idx + j)
            dataset_inputs = [j_p, j_v, j_T]

            # For each joint specified
            for i, urdf_node_name in enumerate(self.get_joint_node_sorted_order()):

                # Find the index of this particular node
                node_index = self.urdf_name_to_graph_index[urdf_node_name]

                # For each variable to use
                for k in variables_to_use:

                    # Add the features to x matrix
                    x[node_index, k*self.history_length+j] = dataset_inputs[k][i]

        # Create the edge matrix
        self.edge_matrix = self.robotGraph.get_edge_index_matrix()
        self.edge_matrix_tensor = torch.tensor(self.edge_matrix,
                                               dtype=torch.long)

        # Create the labels
        lin_acc, ang_vel, j_p, j_v, j_T, f_p, f_v, labels = self.load_data_sorted(
            self.first_index + idx + self.history_length - 1)
        y = torch.tensor(labels, dtype=torch.float64)

        # Create the graph
        graph = Data(x=x, edge_index=self.edge_matrix_tensor,
                     y=y, num_nodes=self.robotGraph.get_num_nodes())
        return graph
    
    def get_helper_heterogeneous_gnn(self, idx):
        """
        Get a dataset entry if we are using a Heterogeneous GNN model.
        """

        # Create the Heterogeneous Data object
        data = HeteroData()

        # Get the edge matrices
        bj, jb, jj, fj, jf = self.robotGraph.get_edge_index_matrices()
        data['base', 'connect',
               'joint'].edge_index = torch.tensor(bj, dtype=torch.long)
        data['joint', 'connect',
             'base'].edge_index = torch.tensor(jb, dtype=torch.long)
        data['joint', 'connect',
             'joint'].edge_index = torch.tensor(jj, dtype=torch.long)
        data['foot', 'connect',
             'joint'].edge_index = torch.tensor(fj, dtype=torch.long)
        data['joint', 'connect',
             'foot'].edge_index = torch.tensor(jf, dtype=torch.long)

        # Set the edge attributes
        bj_attr, jb_attr, jj_attr, fj_attr, jf_attr = self.robotGraph.get_edge_attr_matrices()
        data['base', 'connect',
               'joint'].edge_attr = torch.tensor(bj_attr, dtype=torch.float64)
        data['joint', 'connect',
             'base'].edge_attr = torch.tensor(jb_attr, dtype=torch.float64)
        data['joint', 'connect',
             'joint'].edge_attr = torch.tensor(jj_attr, dtype=torch.float64)
        data['foot', 'connect',
             'joint'].edge_attr = torch.tensor(fj_attr, dtype=torch.float64)
        data['joint', 'connect',
             'foot'].edge_attr = torch.tensor(jf_attr, dtype=torch.float64)

        # Save the labels and number of nodes
        lin_acc, ang_vel, j_p, j_v, j_T, f_p, f_v, labels = self.load_data_sorted(
            self.first_index + idx + self.history_length - 1)
        data.y = torch.tensor(labels, dtype=torch.float64)
        data.num_nodes = self.robotGraph.get_num_nodes()

        # Find out which variables are available
        lin_acc, ang_vel, j_p, j_v, j_T, f_p, f_v, labels = self.load_data_sorted(self.first_index + idx)
        base_variables_to_use = self.find_variables_to_use([lin_acc, ang_vel])
        joint_variables_to_use = self.find_variables_to_use([j_p, j_v, j_T])
        foot_variables_to_use = self.find_variables_to_use([f_p, f_v])
        if len(base_variables_to_use) + len(joint_variables_to_use) + len(foot_variables_to_use) == 0:
            raise ValueError("Dataset must provide at least one input.")
        
        # Calculate the size of the feature matrices
        number_nodes = self.robotGraph.get_num_of_each_node_type()
        base_width = len(base_variables_to_use) * 3 * self.history_length
        joint_width = len(joint_variables_to_use) * self.history_length
        foot_width = len(foot_variables_to_use) * self.history_length
        if base_width <= 0: base_width = 1
        if joint_width <= 0: joint_width = 1
        if foot_width <= 0: foot_width = 1

        # Create the feature matrices
        base_x = torch.ones((number_nodes[0], base_width), dtype=torch.float64)
        joint_x = torch.ones((number_nodes[1], joint_width), dtype=torch.float64)
        foot_x = torch.ones((number_nodes[2], foot_width), dtype=torch.float64)

        # For each dataset entry we include in the history
        for j in range(0, self.history_length):
            # Load the data for this entry
            lin_acc, ang_vel, j_p, j_v, j_T, f_p, f_v, labels = self.load_data_sorted(self.first_index + idx + j)

            # For each joint specified
            joint_data = [j_p, j_v, j_T]
            for i, urdf_node_name in enumerate(self.get_joint_node_sorted_order()):
                node_index = self.urdf_name_to_graph_index[urdf_node_name]

                # For each variable to use
                for k in joint_variables_to_use:
                    joint_x[node_index, k*self.history_length+j] = joint_data[k][i]

            # For each base specified (should be 1)
            base_data = [lin_acc, ang_vel]
            for i, urdf_node_name in enumerate(self.get_base_node_sorted_order()):
                node_index = self.urdf_name_to_graph_index[urdf_node_name]

                # For each variable to use
                for k in base_variables_to_use:
                    base_x[node_index, ((k*3)+0)*self.history_length+j] = base_data[k][i*3]
                    base_x[node_index, ((k*3)+1)*self.history_length+j] = base_data[k][i*3+1]
                    base_x[node_index, ((k*3)+2)*self.history_length+j] = base_data[k][i*3+2]

            # For each foot specified
            foot_data = [f_p, f_v]
            for i, urdf_node_name in enumerate(self.get_foot_node_sorted_order()):
                node_index = self.urdf_name_to_graph_index[urdf_node_name]

                # For each variable to use
                for k in foot_variables_to_use:
                    foot_x[node_index, k*self.history_length+j] = foot_data[k][i]

        # Save the matrices into the HeteroData object
        data['base'].x = base_x
        data['joint'].x = joint_x
        data['foot'].x = foot_x
        return data

    def find_variables_to_use(self, variables_to_check: list[list]) -> np.array:
        """
        This helper method takes in a list, and checks each entry. Each entry
        that isn't None gets its correponding index appended to the return
        value. This allows methods to know which variables are available for
        use from the load_data_sorted() method.
        """

        variables_to_use = []
        for i, variable in enumerate(variables_to_check):
            if variable is not None:
                variables_to_use.append(i)
        variables_to_use = np.array(variables_to_use)
        return variables_to_use