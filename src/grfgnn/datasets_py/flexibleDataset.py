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
        if self.data_format != 'mlp' and self.data_format != 'heterogeneous_gnn':
            raise ValueError(
                "Parameter 'data_format' must be 'mlp' or 'heterogeneous_gnn'."
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

            # Check to make sure that this dataset id matches what we expect.
            # Protects against users passing a folder path to a different
            # dataset sequence, causing a different dataset to be used than
            # expected.
            if self.get_google_drive_file_id() != data[2]:
                raise ValueError("'root' parameter points to a Dataset sequence that doesn't match this Dataset class. Either fix the path to point to the correct sequence, or delete the data in the folder so that the proper sequence can be downloaded.")

        # Parse the robot graph from the URDF file
        if self.data_format == 'heterogeneous_gnn':
            self.robotGraph = HeterogeneousRobotGraph(urdf_path,
                                                      ros_builtin_path,
                                                      urdf_to_desc_path)
        else:
            self.robotGraph = NormalRobotGraph(urdf_path, ros_builtin_path,
                                               urdf_to_desc_path)
            
            # Create second one so that MLP can order outputs to match
            # the URDF file.
            self.robotGraphHetero = HeterogeneousRobotGraph(urdf_path,
                                                      ros_builtin_path,
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
        self.joint_node_indices_sorted = np.array(self.joint_node_indices_sorted, dtype=np.uint)
        self.foot_node_indices_sorted = []
        for urdf_node_name in self.get_foot_node_sorted_order():
            self.foot_node_indices_sorted.append(self.get_urdf_name_to_dataset_array_index()[urdf_node_name])
        self.foot_node_indices_sorted = np.array(self.foot_node_indices_sorted, dtype=np.uint)

        # Calculate means and stds for input and label standardization later
        # self.calculate_mean_and_std()

        # Precompute values for variables_to_use
        lin_acc, ang_vel, j_p, j_v, j_T, f_p, f_v, labels, r_p, r_o = self.load_data_sorted(self.first_index)
        self.variables_to_use_all = self.find_variables_to_use([lin_acc, ang_vel, j_p, j_v, j_T, f_p, f_v])
        self.variables_to_use_base = self.find_variables_to_use([lin_acc, ang_vel])
        self.variables_to_use_joint = self.find_variables_to_use([j_p, j_v, j_T])
        self.variables_to_use_foot = self.find_variables_to_use([f_p, f_v])

        # Check we have necessary variables for some models
        if len(self.variables_to_use_joint) == 0 and self.data_format == 'gnn':
            raise ValueError("Dataset must provide at least one joint input for GNN models.")
        if len(self.variables_to_use_base) + len(self.variables_to_use_joint) + len(self.variables_to_use_foot) == 0:
            raise ValueError("Dataset must provide at least one input.")

        # Premake the tensors for edge attributes and connections for HGNN
        if self.data_format == 'heterogeneous_gnn':
            bj, jb, jj, fj, jf = self.robotGraph.get_edge_index_matrices()
            self.bj = torch.tensor(bj, dtype=torch.long)
            self.jb = torch.tensor(jb, dtype=torch.long)
            self.jj = torch.tensor(jj, dtype=torch.long)
            self.fj = torch.tensor(fj, dtype=torch.long)
            self.jf = torch.tensor(jf, dtype=torch.long)

            bj_attr, jb_attr, jj_attr, fj_attr, jf_attr = self.robotGraph.get_edge_attr_matrices()
            self.bj_attr = torch.tensor(bj_attr, dtype=torch.float64)
            self.jb_attr = torch.tensor(jb_attr, dtype=torch.float64)
            self.jj_attr = torch.tensor(jj_attr, dtype=torch.float64)
            self.fj_attr = torch.tensor(fj_attr, dtype=torch.float64)
            self.jf_attr = torch.tensor(jf_attr, dtype=torch.float64)

        # Precompute feature matrix sizes for HGNN
        # Calculate the size of the feature matrices
        if self.data_format == 'heterogeneous_gnn':
            self.hgnn_number_nodes = self.robotGraph.get_num_of_each_node_type()
            self.base_width = len(self.variables_to_use_base) * 3 * self.history_length
            self.joint_width = len(self.variables_to_use_joint) * self.history_length
            self.foot_width = len(self.variables_to_use_foot) * 3 * self.history_length
            if self.base_width <= 0: self.base_width = 1
            if self.joint_width <= 0: self.joint_width = 1
            if self.foot_width <= 0: self.foot_width = 1

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

        Additionally, used to check already downloaded datasets to
        make sure the user didn't accidentally pass a path to
        a different sequence.
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

        Data must be saved into text files, in the order
        returned by load_data_at_dataset_seq().
        """
        raise self.notImplementedError
    
    # ================================================================
    # ============= DATA SORTING ORDER AND MAPPINGS ==================
    # ================================================================

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
    # ====================== STANDARDIZATION =========================
    # ================================================================
    def calculate_mean_and_std(self):
        """
        This helper method calculates the mean and std of
        the dataset data per feature, for use in standardization
        later.
        """

        # Calculate the number of features we need to get mean and std for.
        num_features = 0
        lin_acc, ang_vel, j_p, j_v, j_T, f_p, f_v, labels = self.load_data_at_dataset_seq(0)
        to_check = [lin_acc, ang_vel, j_p, j_v, j_T, f_p, f_v, labels]        
        for array in to_check:
            if array is not None:
                num_features += len(array)

        # Calculate the line_num and line_index for each feature
        file_data_array = np.zeros((2, num_features), dtype=np.int64)
        line_i = 0
        curr_feature_i = 0
        for array in to_check:
            if array is not None:
                line_index_i = 0
                for val in array:
                    file_data_array[0,curr_feature_i] = line_i
                    file_data_array[1,curr_feature_i] = line_index_i
                    line_index_i += 1
                    curr_feature_i += 1
                line_i += 1

        print(file_data_array)

        # For each feature
        self.means = np.zeros((num_features))
        self.stds = np.zeros((num_features))
        vfunc = np.vectorize(self.load_single_value)
        for i in range(0, num_features):
            # Get every occurance of that feature
            vals: np.array = vfunc(range(0, self.len()), file_data_array[0,i], file_data_array[1,i])
            print(vals)

            # Calculate mean and std
            self.means[i] = np.mean(vals)
            self.stds[i] = np.std(vals)

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

        Next, labels are checked to make sure they aren't None.

        Parameters:
            seq_num (int): The sequence number of the txt file
                whose data should be loaded.

        Returns:
            Same values as load_data_at_dataset_seq(), but order of
            values inside arrays have been sorted.
        """
        lin_acc, ang_vel, j_p, j_v, j_T, f_p, f_v, labels, r_p, r_o = self.load_data_at_dataset_seq(seq_num)

        # Sort the joint information
        unsorted_list = [j_p, j_v, j_T]
        sorted_list = []
        for unsorted_array in unsorted_list:
            if unsorted_array is not None:
                sorted_list.append(unsorted_array[self.joint_node_indices_sorted])
            else:
                sorted_list.append(None)

        # Sort the foot information
        unsorted_foot_list = [f_p, f_v]
        sorted_foot_list = []
        for unsorted_array in unsorted_foot_list:
            if unsorted_array is not None:
                sorted_array = []
                for index in self.foot_node_indices_sorted:
                    for i in range(0, 3):
                        sorted_array.append(unsorted_array[int(index*3+i)])
                sorted_foot_list.append(sorted_array)
            else:
                sorted_foot_list.append(None)

        # Sort the ground truth labels
        labels_sorted = None
        if labels is None:
            raise ValueError("Dataset must provide labels.")
        else:
            labels_sorted = labels[self.foot_node_indices_sorted]

        return lin_acc, ang_vel, sorted_list[0], sorted_list[1], sorted_list[2], sorted_foot_list[0], sorted_foot_list[1], labels_sorted, r_p, r_o

    def load_single_value(self, seq_num: int, line_num: int, line_index: int):
        """
        This helper method loads a single value from the txt file
        at "seq_num", using the given line_num and line_index.
        """

        with open(str(Path(self.processed_dir, str(seq_num) + ".txt")), 'r') as f:
            line = None
            for i in range(0, line_num+1):
                line = f.readline().split(" ")[:-1]
            return float(line[line_index])

    def load_data_at_dataset_seq(self, seq_num: int):
        """
        This helper function opens the txt file at "seq_num"
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
            r_p (np.array) - Robot position (GT), in the order (x, y, z).
            r_o (np.array) - Robot orientation (GT) as a quaternion, in the order (x, y, z, w).

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
        if self.data_format == 'mlp':
            return self.get_helper_mlp(idx)
        elif self.data_format == 'heterogeneous_gnn':
            return self.get_helper_heterogeneous_gnn(idx)
        
    def get_helper_mlp(self, idx: int):
        """
        Gets a Dataset entry if we are using an MLP model.
        """

        # Load the dataset information
        x = None
        for i in range(0, self.history_length):

            # Only add variables that aren't None
            lin_acc, ang_vel, j_p, j_v, j_T, f_p, f_v, labels, r_p, r_o = self.load_data_sorted(self.first_index + idx + i)
            dataset_inputs = [lin_acc, ang_vel, j_p, j_v, j_T, f_p, f_v]
            final_input = np.array([])
            for y in self.variables_to_use_all:
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
        lin_acc, ang_vel, j_p, j_v, j_T, f_p, f_v, labels, r_p, r_o = self.load_data_sorted(self.first_index + idx + self.history_length - 1)
        y = torch.ones((4), dtype=torch.float64)
        urdf_name_to_hetero_graph_index = self.robotGraphHetero.get_node_name_to_index_dict()
        for i, urdf_node_name in enumerate(self.get_foot_node_sorted_order()):

                # Find the index of this particular node
                node_index = urdf_name_to_hetero_graph_index[urdf_node_name]

                # Add the label feature
                y[node_index] = labels[i]

        return x, y
    
    def get_helper_heterogeneous_gnn(self, idx):
        """
        Get a dataset entry if we are using a Heterogeneous GNN model.
        """

        # Create the Heterogeneous Data object
        data = HeteroData()

        # Get the edge matrices
        data['base', 'connect','joint'].edge_index = self.bj
        data['joint', 'connect','base'].edge_index = self.jb
        data['joint', 'connect','joint'].edge_index = self.jj
        data['foot', 'connect','joint'].edge_index = self.fj
        data['joint', 'connect','foot'].edge_index = self.jf

        # Set the edge attributes
        data['base', 'connect','joint'].edge_attr = self.bj_attr
        data['joint', 'connect','base'].edge_attr = self.jb_attr
        data['joint', 'connect','joint'].edge_attr = self.jj_attr
        data['foot', 'connect','joint'].edge_attr = self.fj_attr
        data['joint', 'connect','foot'].edge_attr = self.jf_attr

        # Save the number of nodes
        data.num_nodes = self.robotGraph.get_num_nodes()

        # Make the labels
        lin_acc, ang_vel, j_p, j_v, j_T, f_p, f_v, labels, r_p, r_o = self.load_data_sorted(
            self.first_index + idx + self.history_length - 1)
        data.y = torch.ones((4), dtype=torch.float64)
        for i, urdf_node_name in enumerate(self.get_foot_node_sorted_order()):

                # Find the index of this particular node
                node_index = self.urdf_name_to_graph_index[urdf_node_name]

                # Add the label feature
                data.y[node_index] = labels[i]

        # Create the feature matrices
        base_x = torch.ones((self.hgnn_number_nodes[0], self.base_width), dtype=torch.float64)
        joint_x = torch.ones((self.hgnn_number_nodes[1], self.joint_width), dtype=torch.float64)
        foot_x = torch.ones((self.hgnn_number_nodes[2], self.foot_width), dtype=torch.float64)

        # For each dataset entry we include in the history
        for j in range(0, self.history_length):
            # Load the data for this entry
            lin_acc, ang_vel, j_p, j_v, j_T, f_p, f_v, labels, r_p, r_o = self.load_data_sorted(self.first_index + idx + j)

            # For each joint specified
            joint_data = [j_p, j_v, j_T]
            for i, urdf_node_name in enumerate(self.get_joint_node_sorted_order()):
                node_index = self.urdf_name_to_graph_index[urdf_node_name]

                # For each variable to use
                for k in self.variables_to_use_joint:
                    joint_x[node_index, k*self.history_length+j] = joint_data[k][i]

            # For each base specified (should be 1)
            base_data = [lin_acc, ang_vel]
            for i, urdf_node_name in enumerate(self.get_base_node_sorted_order()):
                node_index = self.urdf_name_to_graph_index[urdf_node_name]

                # For each variable to use
                for k in self.variables_to_use_base:
                    base_x[node_index, ((k*3)+0)*self.history_length+j] = base_data[k][i*3]
                    base_x[node_index, ((k*3)+1)*self.history_length+j] = base_data[k][i*3+1]
                    base_x[node_index, ((k*3)+2)*self.history_length+j] = base_data[k][i*3+2]

            # For each foot specified
            foot_data = [f_p, f_v]
            for i, urdf_node_name in enumerate(self.get_foot_node_sorted_order()):
                node_index = self.urdf_name_to_graph_index[urdf_node_name]

                # For each variable to use
                for k in self.variables_to_use_foot:
                    foot_x[node_index, ((k*3)+0)*self.history_length+j] = foot_data[k][i*3]
                    foot_x[node_index, ((k*3)+1)*self.history_length+j] = foot_data[k][i*3+1]
                    foot_x[node_index, ((k*3)+2)*self.history_length+j] = foot_data[k][i*3+2]

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