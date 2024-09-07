import torch
from torch_geometric.data import Dataset, HeteroData
from ..graphParser import NormalRobotGraph, HeterogeneousRobotGraph
from pathlib import Path
from torchvision.datasets.utils import download_file_from_google_drive
import scipy.io as sio
import numpy as np
import urllib.request

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
                 data_format: str = 'heterogeneous_gnn',
                 history_length: int = 1,
                 normalize: bool = False,
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
                data_format (str): Either 'dynamics', 'mlp', or 'heterogeneous_gnn'. 
                    Determines how the get() method returns data.
                history_length (int): The length of the history of inputs to use
                    for our graph entries.
                normalize (bool): If true, each individual 'Data' or 'HeteroData' 
                    object will return each of the node inputs normalized across the
                    time domain, specifically only for the `history_length` considered
                    in the current entry. The normalization is unique per node input, and 
                    unique per specific dataset entry.
        """
        # Check for valid data format
        self.data_format = data_format
        if self.data_format != 'dynamics' and self.data_format != 'mlp' and self.data_format != 'heterogeneous_gnn':
            raise ValueError(
                "Parameter 'data_format' must be 'dynamics', 'mlp', or 'heterogeneous_gnn'."
            )

        # Setup the directories for raw and processed data, download it, 
        # and process
        self.root = root
        super().__init__(str(root), transform, pre_transform, pre_filter)

        # Load the data from the mat file
        path_to_mat = Path(self.root, 'processed', 'data.mat')
        self.mat_data = sio.loadmat(path_to_mat)

        # Get the first index and length of dataset
        info_path = Path(root, 'processed', 'info.txt')
        with open(info_path, 'r') as f:
            data = f.readline().split(" ")
            
            self.history_length = history_length
            if self.history_length != 1 and self.data_format == 'dynamics':
                raise ValueError("Data format of 'dynamics' only supports history length of 1.")

            self.length = int(data[0]) - self.history_length + 1
            if self.data_format == 'dynamics':
                self.length -= 2 # We can't use the first or last entry, due to the derivative.
            if self.length <= 0:
                raise ValueError(
                    "Dataset has too few entries for the provided 'history_length'."
                )

            # Check to make sure that this dataset id matches what we expect.
            # Protects against users passing a folder path to a different
            # dataset sequence, causing a different dataset to be used than
            # expected.
            file_id, loc = self.get_file_id_and_loc()
            if file_id != data[1]:
                raise ValueError("'root' parameter points to a Dataset sequence that doesn't match this Dataset class. Either fix the path to point to the correct sequence, or delete the data in the folder so that the proper sequence can be downloaded.")

        # Parse the robot graph from the URDF file
        self.robotGraph = HeterogeneousRobotGraph(urdf_path, ros_builtin_path,
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


        # Define the order that the sorted joint and foot data should be in
        self.urdf_name_to_graph_index_foot = self.robotGraph.get_node_name_to_index_dict_for_type('foot')
        self.urdf_name_to_graph_index_joint = self.robotGraph.get_node_name_to_index_dict_for_type('joint')
        self.urdf_name_to_graph_index_base = self.robotGraph.get_node_name_to_index_dict_for_type('base')

        # Precompute the array indexes matching the defined orders for joints and feet
        self.foot_node_indices_sorted = np.zeros((len(self.urdf_name_to_graph_index_foot)), dtype=np.float64)
        for urdf_node_name in self.urdf_name_to_graph_index_foot.keys():
            graph_node_index = self.urdf_name_to_graph_index_foot[urdf_node_name]
            self.foot_node_indices_sorted[graph_node_index] = self.get_urdf_name_to_dataset_array_index()[urdf_node_name]
        self.foot_node_indices_sorted = np.array(self.foot_node_indices_sorted, dtype=np.uint)

        self.joint_node_indices_sorted = np.zeros((len(self.urdf_name_to_graph_index_joint)), dtype=np.float64)
        for urdf_node_name in self.urdf_name_to_graph_index_joint.keys():
            graph_node_index = self.urdf_name_to_graph_index_joint[urdf_node_name]
            self.joint_node_indices_sorted[graph_node_index] = self.get_urdf_name_to_dataset_array_index()[urdf_node_name]
        self.joint_node_indices_sorted = np.array(self.joint_node_indices_sorted, dtype=np.uint)

        self.base_node_indices_sorted = np.zeros((len(self.urdf_name_to_graph_index_base)), dtype=np.float64)
        for urdf_node_name in self.urdf_name_to_graph_index_base.keys():
            graph_node_index = self.urdf_name_to_graph_index_base[urdf_node_name]
            self.base_node_indices_sorted[graph_node_index] = self.get_urdf_name_to_dataset_array_index()[urdf_node_name]
        self.base_node_indices_sorted = np.array(self.joint_node_indices_sorted, dtype=np.uint)

        # Set normalize parameter for use later
        self.normalize = normalize

        # Precompute values for variables_to_use
        lin_acc, ang_vel, j_p, j_v, j_T, f_p, f_v, labels, r_p, r_o, timestamps = self.load_data_sorted(0)
        self.variables_to_use_all = self.find_variables_to_use([lin_acc, ang_vel, j_p, j_v, j_T, f_p, f_v])
        self.variables_to_use_base = self.find_variables_to_use([lin_acc, ang_vel])
        self.variables_to_use_joint = self.find_variables_to_use([j_p, j_v, j_T])
        self.variables_to_use_foot = self.find_variables_to_use([f_p, f_v])

        # Check we have necessary variables for some models
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
        file_id, loc = self.get_file_id_and_loc()
        if loc == "Google":
            download_file_from_google_drive(file_id,
                                            Path(self.root, 'raw'),
                                            self.get_downloaded_dataset_file_name())
        elif loc == "Dropbox":
            urllib.request.urlretrieve(file_id, Path(self.root, "raw", self.get_downloaded_dataset_file_name()))
        else:
            raise NotImplementedError("Only Google and Dropbox are implemented.")
        
    def get_file_id_and_loc(self):
        """
        Method for child classes to choose which sequence to load;
        used if the dataset is downloaded.

        Additionally, used to check already downloaded datasets to
        make sure the user didn't accidentally pass a path to
        a different sequence.

        Returns:
            file_id (str) - File id (or link) for download
            location (str) - Either "Google" or "Dropbox"
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
        return ["data.mat", "info.txt"]

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
    
    def get_urdf_name_to_dataset_array_index(self) -> dict:
        """
        Returns a dictionary that maps strings of
        URDF joint names to the corresponding index
        in the dataset array. This allows us to know
        which dataset entries correspond to the which 
        joints in the URDF.
        """
        raise self.notImplementedError
    
    def urdf_to_pin_order_mapping(self):
        """
        Returns the mappings from the URDF order to
        the pinocchio model order.

        Returns:
            joint_mappings (np.array) - An array that maps URDF joint order to Pin joint order, of shape [12]
            foot_mappings (np.array) - And array that maps URDF foot order to Pin foot order, of shape [4]
        """
        raise self.notImplementedError
    
    def pin_to_urdf_order_mapping(self):
        """
        Returns the mappings from the pinocchio model 
        order to the URDF order. Passed to the Dynamics
        Model so that it can return the outputs in URDF order.

        Returns:
            joint_mappings (np.array) - An array that maps Pin joint order to URDF joint order, of shape [12]
            foot_mappings (np.array) - An array that maps Pin foot order to URDF foot order, of shape [4]
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
    
    def get_data_metadata(self) -> tuple:
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
        However, the joint and feet are sorted so that they match 
        the order in the URDF file. Additionally, the foot labels 
        are sorted so it matches the order in the URDF file.

        Next, labels are checked to make sure they aren't None. 
        Finally, normalize the data if self.normalize was set as True.
        We calculate the standard deviation for this normalization 
        using Bessel's correction (n-1 used instead of n).

        Parameters:
            seq_num (int): The sequence number of the txt file
                whose data should be loaded.

        Returns:
            Same values as load_data_at_dataset_seq(), but order of
            values inside arrays have been sorted (and potentially
            normalized).
        """
        lin_acc, ang_vel, j_p, j_v, j_T, f_p, f_v, labels, r_p, r_o, timestamps = self.load_data_at_dataset_seq(seq_num)

        # Sort the joint information
        unsorted_list = [j_p, j_v, j_T]
        sorted_list = []
        for unsorted_array in unsorted_list:
            if unsorted_array is not None:
                sorted_list.append(unsorted_array[:,self.joint_node_indices_sorted])
            else:
                sorted_list.append(None)

        # Sort the foot information
        unsorted_foot_list = [f_p, f_v]
        sorted_foot_list = []
        for unsorted_array in unsorted_foot_list:
            if unsorted_array is not None:
                sorted_indices = []
                for index in self.foot_node_indices_sorted:
                    for i in range(0, 3):
                        sorted_indices.append(int(index*3+i))
                sorted_foot_list.append(unsorted_array[:,sorted_indices])
            else:
                sorted_foot_list.append(None)

        # Sort the ground truth labels
        labels_sorted = None
        if labels is None:
            raise ValueError("Dataset must provide labels.")
        else:
            labels_sorted = labels[self.foot_node_indices_sorted]

        # Normalize the data if desired
        norm_arrs = [None, None, None, None, None, None, None, None, None]
        if self.normalize:
            # Normalize all data except the labels
            to_normalize_array = [lin_acc, ang_vel, sorted_list[0], sorted_list[1], sorted_list[2], sorted_foot_list[0], sorted_foot_list[1], r_p, r_o]
            for i, array in enumerate(to_normalize_array):
                if (array is not None) and (array.shape[0] > 1):
                    array_tensor = torch.from_numpy(array)
                    norm_arrs[i] = np.nan_to_num((array_tensor-torch.mean(array_tensor,axis=0))/torch.std(array_tensor, axis=0, correction=1).numpy(), copy=False, nan=0.0)

            return norm_arrs[0], norm_arrs[1], norm_arrs[2], norm_arrs[3], norm_arrs[4], norm_arrs[5], norm_arrs[6], labels_sorted, norm_arrs[7], norm_arrs[8], timestamps
        else:
            return lin_acc, ang_vel, sorted_list[0], sorted_list[1], sorted_list[2], sorted_foot_list[0], sorted_foot_list[1], labels_sorted, r_p, r_o, timestamps

    def load_data_at_dataset_seq(self, seq_num: int):
        """
        This helper function opens the txt file at "seq_num"
        and loads dataset information for that sequence.

        NOTE: even if self.normalize is set, this WON'T return
        normalized data. Use load_data_sorted() for that.

        Parameters:
            seq_num (int): The sequence number of the txt file
                whose data should be loaded.

        Returns:
            lin_acc (np.array) - IMU linear acceleration, of shape [history_length, 3]
            ang_vel (np.array) - IMU angular velocity, of shape [history_length, 3]
            j_p (np.array) - Joint positions, of shape [history_length, 12]
            j_v (np.array) - Joint velocities, of shape [history_length, 12]
            j_T (np.array) - Joint Torques, of shape [history_length, 12]
            f_p (np.array) - Foot position, of shape [history_length, 12]
            f_v (np.array) - Foot velocity, of shape [history_length, 12]
            labels (np.array) - The Dataset labels (either Z direction GRFs, or contact states), of shape [4]
                Only the latest entry in the time window is used as the labels.
            r_p (np.array) - Robot position (GT), in the order (x, y, z), of shape [history_length, 3]
            r_o (np.array) - Robot orientation (GT) as a quaternion, in the order (x, y, z, w), of shape [history_length, 4]
            timestamps (np.array) - Array containing the timestamps of the data. The rows
                correspond to the history length and the columns correspond to the specific
                timestamp per data. Column 0 contains the grf_timestamp (for the GRF labels),
                Column 1 contains the joint_timestamp (for the joint data, and additionally
                the robot pose information), and Column 2 contains the imu_timestamp (for the
                linear acceleration and angular velocity of the IMU). This value is in the 
                shape of (history_length, 3).

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
        if self.data_format == 'dynamics':
            return self.get_helper_dynamics(idx)
        if self.data_format == 'mlp':
            return self.get_helper_mlp(idx)
        elif self.data_format == 'heterogeneous_gnn':
            return self.get_helper_heterogeneous_gnn(idx)
        
    def get_helper_dynamics(self, idx: int):
        """
        Gets a dataset entry for the dynamics model. Shifts the idx internally
        by 1 so that we can calculate the derivative of certain variables.

        Also, reorders from URDF order to pinnochio model order, including
        the labels.
        
        Returns:
            q - The generalized coordinates
            vel - The generalized velocity
            acc - The generalized acceleration
            tau - The generalized torques
            labels - Ground Truth labels (in pinocchio foot order)
        """
        _,  ang_vel_prev,    _, j_v_prev,   _,   _,   _,      _, r_p_prev,   _, ts_prev = self.load_data_sorted(idx)
        lin_acc, ang_vel,  j_p,      j_v, j_T, f_p, f_v, labels,      r_p, r_o,      ts = self.load_data_sorted(idx + 1)
        _,  ang_vel_next,    _, j_v_next,   _,   _,   _,      _, r_p_next,   _, ts_next = self.load_data_sorted(idx + 2)

        # load_data_sorted returns data sorted by URDF order.
        # However, we need it in the pinnocchio model order.
        pin_sorted_order_joints, pin_sorted_order_labels = self.urdf_to_pin_order_mapping()
        j_p = j_p[:, pin_sorted_order_joints]
        j_v_prev = j_v_prev[:, pin_sorted_order_joints]
        j_v = j_v[:, pin_sorted_order_joints]
        j_v_next = j_v_next[:, pin_sorted_order_joints]     
        j_T = j_T[:, pin_sorted_order_joints] 
        labels = labels[pin_sorted_order_labels]

        # Calculate delta_t
        delta_t_imu = ts_next[:, 2] - ts_prev[:, 2]
        delta_t_joint = ts_next[:, 1] - ts_prev[:, 1]

        # Calculate linear velocity, angular acceleration, and joint acceleration
        lin_vel = (r_p_next - r_p_prev) / delta_t_joint
        ang_acc = (ang_vel_next - ang_vel_prev) / delta_t_imu
        j_a = (j_v_next - j_v_prev) / delta_t_joint

        # Create the generalized coordinate vectors
        q = torch.tensor((np.hstack((np.hstack((r_p, r_o)), j_p))[0]), dtype=torch.float64)
        vel = torch.tensor((np.hstack((np.hstack((lin_vel, ang_vel)), j_v))[0]), dtype=torch.float64)
        acc = torch.tensor((np.hstack((np.hstack((lin_acc, ang_acc)), j_a))[0]), dtype=torch.float64)
        tau = torch.tensor((np.hstack((np.zeros([1, 6]), j_T))[0]), dtype=torch.float64)
        labels = torch.tensor((labels), dtype=torch.float64)

        return q, vel, acc, tau, labels

    def get_helper_mlp(self, idx: int):
        """
        Gets a Dataset entry if we are using an MLP model.
        """

        # Load the dataset information
        x = None

        # Only add variables that aren't None
        lin_acc, ang_vel, j_p, j_v, j_T, f_p, f_v, labels, r_p, r_o, timestamps = self.load_data_sorted(idx)
        dataset_inputs = [lin_acc, ang_vel, j_p, j_v, j_T, f_p, f_v]
        final_input = np.array([])
        for y in self.variables_to_use_all:
            final_input = np.concatenate((final_input, dataset_inputs[y].flatten('F')), axis=0) 

        # Construct the input tensor
        x = torch.tensor((final_input), 
                                dtype=torch.float64).unsqueeze(0)

        # Flatten the tensor if necessary
        if len(x.size()) > 1:
            x = torch.flatten(torch.transpose(x, 0, 1), 0, 1)

        # Create the ground truth labels
        y = torch.tensor(labels, dtype=torch.float64)
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

        # Create the feature matrices
        base_x = torch.ones((self.hgnn_number_nodes[0], self.base_width), dtype=torch.float64)
        joint_x = torch.ones((self.hgnn_number_nodes[1], self.joint_width), dtype=torch.float64)
        foot_x = torch.ones((self.hgnn_number_nodes[2], self.foot_width), dtype=torch.float64)

        # Load the data for this entry
        lin_acc, ang_vel, j_p, j_v, j_T, f_p, f_v, labels, r_p, r_o, timestamps = self.load_data_sorted(idx)

        # For each joint specified
        joint_data = [j_p, j_v, j_T]
        for i, urdf_node_name in enumerate(self.urdf_name_to_graph_index_joint.keys()):
            # For each variable to use
            final_input = torch.ones((0), dtype=torch.float64)
            for k in self.variables_to_use_joint:
                final_input = torch.cat((final_input, torch.tensor(joint_data[k][:,i].flatten('F'), dtype=torch.float64)), axis=0)

            joint_x[i] = final_input

        # For each base specified (should be 1)
        base_data = [lin_acc, ang_vel]
        for i, urdf_node_name in enumerate(self.urdf_name_to_graph_index_base.keys()):
            # For each variable to use
            final_input = torch.ones((0), dtype=torch.float64)
            for k in self.variables_to_use_base:
                final_input = torch.cat((final_input, torch.tensor(base_data[k][:,i:i+3].flatten('F'), dtype=torch.float64)), axis=0)
            base_x[i] = final_input

        # For each foot specified
        foot_data = [f_p, f_v]
        for i, urdf_node_name in enumerate(self.urdf_name_to_graph_index_foot.keys()):
            # For each variable to use
            final_input = torch.ones((0), dtype=torch.float64)
            for k in self.variables_to_use_foot:
                final_input = torch.cat((final_input, torch.tensor(foot_data[k][:,(3*i):(3*i)+3].flatten('F'), dtype=torch.float64)), axis=0)
            if final_input.shape[0] != 0:
                foot_x[i] = final_input
        
        # Make the labels
        data.y = torch.tensor(labels, dtype=torch.float64)

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