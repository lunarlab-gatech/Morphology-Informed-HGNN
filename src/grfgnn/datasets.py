import torch
import torch_geometric
from torch_geometric.data import Data, Dataset, HeteroData
from .graphParser import RobotGraph, NormalRobotGraph, HeterogeneousRobotGraph
import networkx
import numpy as np
import matplotlib.pyplot as plt
from rosbags.highlevel import AnyReader
from pathlib import Path
from itertools import islice
import os
from torchvision.datasets.utils import download_file_from_google_drive
from rosbags.rosbag2 import Writer
from rosbags.typesys import Stores, get_typestore, get_types_from_msg
from rosbags.interfaces import ConnectionExtRosbag2
from typing import cast


class FlexibleDataset(Dataset):
    """
    Dataset class that can be used for a MLP, normal GNN, or 
    heterogeneous GNN.
    """

    # Used when initalizing the parent class instead of a proper
    # child classs
    notImplementedError = NotImplementedError(
        "Don't call this class directly, but use one of \
        the child classes in order to choose which dataset \
        sequence you want to load.")

    def __init__(self,
                 root: str,
                 urdf_path: Path,
                 ros_builtin_path: str,
                 urdf_to_desc_path: str,
                 data_format: str = 'gnn',
                 transform=None,
                 pre_transform=None,
                 pre_filter=None):
        """
        Parameters:
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
        super().__init__(root, transform, pre_transform, pre_filter)

        # Get the first index and length of dataset
        info_path = Path(root, 'processed', 'info.txt')
        with open(info_path, 'r') as f:
            data = f.readline().split(" ")
            self.length = int(data[0])
            self.first_index = int(data[1])

        # Make sure the URDF file we were given properly matches
        # what we are expecting
        if self.data_format == 'heterogeneous_gnn':
            robotURDF = HeterogeneousRobotGraph(urdf_path, ros_builtin_path,
                                                urdf_to_desc_path, True)
        else:
            robotURDF = NormalRobotGraph(urdf_path, ros_builtin_path,
                                         urdf_to_desc_path, True)
        passed_urdf_name = robotURDF.robot_urdf.name
        expected_name = self.get_expected_urdf_name()
        if passed_urdf_name != expected_name:
            raise ValueError(
                "Invalid URDF: \"" + passed_urdf_name +
                "\". This dataset was collected on the \"" + expected_name +
                "\" robot. Please pass in the URDF file for the \"" +
                expected_name + "\" robot, not for the \"" + passed_urdf_name +
                "\" robot.")

        # Save the URDF info
        self.URDF = robotURDF

        # Get node name to index mapping
        self.urdf_name_to_index = self.URDF.get_node_name_to_index_dict()

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

    def get_ground_truth_label_indices(self):
        """
        This helper method returns the node indices that correspond to 
        the ground truth forces labels contained in the "y" of each 
        Data object. The number x in the ith index tells us that 
        ground truth label i matches up with node x from the GNN output.

        For example, if the graph has 8 nodes, and there are only 4 
        ground truth labels, this method might return [0, 3, 4, 7].
        This means that the first ground truth label should match
        the output of the node at index 0, the second ground truth label
        should match the output of the node at index 3, and so on.

        Returns:
            ground_truth_graph_indices (list[int]) - List of the indices of
            the graph nodes that should output the ground truth labels 
            at the same corresponding index.
        """
        if self.ground_truth_graph_indices is None:
            raise self.notImplementedError
        else:
            return self.ground_truth_graph_indices

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

    def load_data_at_ros_seq(self, ros_seq: int):
        """
        This helper function opens the file named "ros_seq"
        and loads the position, velocity, and effort information.

        Parameters:
            ros_seq (int): The sequence number of the ros message
                whose data should be loaded.
        """
        raise self.notImplementedError

    def get_helper_gnn(self, idx):
        """
        Get a dataset entry if we are using a GNN model.
        """
        raise self.notImplementedError

    def get_helper_mlp(self, idx):
        """
        Gets a Dataset entry if we are using an MLP model.
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


class Go1SimulatedDataset(FlexibleDataset):
    """
    Dataset class for the simulated data for the UniTree Go1.
    """

    def __init__(self,
                 root: str,
                 urdf_path: Path,
                 ros_builtin_path: str,
                 urdf_to_desc_path: str,
                 data_format: str = 'gnn',
                 transform=None,
                 pre_transform=None,
                 pre_filter=None):
        """
        Constructor for Go1 Simulated Dataset provided by Dr. Xiong.
        """
        super().__init__(root, urdf_path, ros_builtin_path, urdf_to_desc_path,
                         data_format, transform, pre_transform, pre_filter)

        # Map urdf names to array indexes
        self.urdf_to_dataset_index = {
            'FR_hip_joint': 0,
            'FR_thigh_joint': 1,
            'FR_calf_joint': 2,
            'FL_hip_joint': 3,
            'FL_thigh_joint': 4,
            'FL_calf_joint': 5,
            'RR_hip_joint': 6,
            'RR_thigh_joint': 7,
            'RR_calf_joint': 8,
            'RL_hip_joint': 9,
            'RL_thigh_joint': 10,
            'RL_calf_joint': 11,
            'FR_foot_fixed': 0,
            'FL_foot_fixed': 1,
            'RR_foot_fixed': 2,
            'RL_foot_fixed': 3,
        }

        # Define the names and indicies that contain ground truth labels
        self.ground_truth_urdf_names = [
            'FR_foot_fixed',
            'FL_foot_fixed',
            'RR_foot_fixed',
            'RL_foot_fixed',
        ]

        self.ground_truth_graph_indices = []
        for urdf_name in self.ground_truth_urdf_names:
            self.ground_truth_graph_indices.append(
                self.urdf_name_to_index[urdf_name])

        self.ground_truth_array_indices = []
        for urdf_name in self.ground_truth_urdf_names:
            self.ground_truth_array_indices.append(
                self.urdf_to_dataset_index[urdf_name])

        # Define the nodes that should recieve features
        self.nodes_for_attributes = [
            'FL_hip_joint', 'FL_thigh_joint', 'FL_calf_joint', 'FR_hip_joint',
            'FR_thigh_joint', 'FR_calf_joint', 'RL_hip_joint',
            'RL_thigh_joint', 'RL_calf_joint', 'RR_hip_joint',
            'RR_thigh_joint', 'RR_calf_joint'
        ]

    def process(self):
        bag_numbers = np.linspace(0, 99, 100)

        # Dataset has no sequence numbers, so hardcode first sequence number
        first_seq = 0
        curr_seq_num = first_seq

        for val in bag_numbers:
            # Set up a reader to read the rosbag
            path_to_bag = os.path.join(
                self.root, 'raw',
                'traj_' + str(int(val)).rjust(4, '0') + '.bag')
            self.reader = AnyReader([Path(path_to_bag)])
            self.reader.open()
            grf_gen = self.reader.messages(connections=[
                x for x in self.reader.connections if x.topic == "grf"
            ])
            pos_gen = self.reader.messages(connections=[
                x for x in self.reader.connections
                if x.topic == "joint_positions"
            ])
            vel_gen = self.reader.messages(connections=[
                x for x in self.reader.connections
                if x.topic == "joint_velocities"
            ])
            tau_gen = self.reader.messages(connections=[
                x for x in self.reader.connections
                if x.topic == "joint_torques"
            ])

            # Extract all of the relevant data into arrays
            grf_data = []
            pos_data = []
            vel_data = []
            tau_data = []

            for connection, _timestamp, rawdata in grf_gen:
                msg = self.reader.deserialize(rawdata, connection.msgtype)
                grf_data.append(msg.data)
            for connection, _timestamp, rawdata in pos_gen:
                msg = self.reader.deserialize(rawdata, connection.msgtype)
                pos_data.append(msg.data)
            for connection, _timestamp, rawdata in vel_gen:
                msg = self.reader.deserialize(rawdata, connection.msgtype)
                vel_data.append(msg.data)
            for connection, _timestamp, rawdata in tau_gen:
                msg = self.reader.deserialize(rawdata, connection.msgtype)
                tau_data.append(msg.data)

            # Make sure we have the same amount of data in each array
            # Otherwise, we have an issue
            if len(grf_data) != len(pos_data) or len(pos_data) != len(vel_data) or \
                len(vel_data) != len(tau_data):
                raise ValueError("Dataset has different amounts of data.")

            # Iterate through the arrays and save important data in txt file
            for i in range(len(grf_data)):
                with open(
                        str(
                            Path(self.processed_dir,
                                 str(curr_seq_num) + ".txt")), "w") as f:
                    arrays = [
                        grf_data[i], pos_data[i], vel_data[i], tau_data[i]
                    ]
                    for array in arrays:
                        for val in array:
                            f.write(str(val) + " ")
                        f.write('\n')
                    curr_seq_num += 1

        # Write a txt file to save the dataset length & and first sequence index
        length = curr_seq_num
        with open(os.path.join(self.processed_dir, "info.txt"), "w") as f:
            f.write(str(length) + " " + str(first_seq))

    def get_expected_urdf_name(self):
        return "go1"

    def get_start_and_end_seq_ids(self):
        return 0, 249099

    def load_data_at_ros_seq(self, ros_seq: int):
        # Open the file with the proper index
        grfs, positions, velocities, torques = [], [], [], []
        with open(os.path.join(self.processed_dir,
                               str(ros_seq) + ".txt"), 'r') as f:
            line = f.readline().split(" ")[:-1]
            for i in range(0, len(line)):
                grfs.append(float(line[i]))
            line = f.readline().split(" ")[:-1]
            for i in range(0, len(line)):
                positions.append(float(line[i]))
            line = f.readline().split(" ")[:-1]
            for i in range(0, len(line)):
                velocities.append(float(line[i]))
            line = f.readline().split(" ")[:-1]
            for i in range(0, len(line)):
                torques.append(float(line[i]))

        # Extract the ground truth labels
        ground_truth_labels = []
        for val in self.ground_truth_array_indices:
            ground_truth_labels.append(grfs[val])

        return positions, velocities, torques, ground_truth_labels

    def get_helper_gnn(self, idx):
        # Load the rosbag information
        positions, velocities, torques, ground_truth_labels = self.load_data_at_ros_seq(
            self.first_index + idx)

        # Create a note feature matrix
        x = torch.ones((self.URDF.get_num_nodes(), 3), dtype=torch.float)

        # For each joint specified
        for urdf_node_name in self.nodes_for_attributes:

            # Find the index of this particular node
            node_index = self.urdf_name_to_index[urdf_node_name]

            # Get the msg array index
            msg_ind = self.urdf_to_dataset_index[urdf_node_name]

            # Add the features to x matrix
            x[node_index, 0] = positions[msg_ind]
            x[node_index, 1] = velocities[msg_ind]
            x[node_index, 2] = torques[msg_ind]

        # Create the graph
        graph = Data(x=x,
                     edge_index=self.edge_matrix_tensor,
                     y=torch.tensor(ground_truth_labels, dtype=torch.float),
                     num_nodes=self.URDF.get_num_nodes())
        return graph

    def get_helper_mlp(self, idx):
        # Load the rosbag information
        positions, velocities, torques, ground_truth_labels = self.load_data_at_ros_seq(
            self.first_index + idx)

        # Make the network inputs
        x = torch.tensor((positions + velocities + torques), dtype=torch.float)

        # Create the ground truth lables
        y = torch.tensor(ground_truth_labels,
                         dtype=torch.float)
        return x, y

    def get_helper_heterogeneous_gnn(self, idx):
        # Create the Heterogeneous Data object
        data = HeteroData()

        # Get the edge matrices
        bj, jb, jj, fj, jf = self.URDF.get_edge_index_matrices()
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

        # Load the rosbag information
        positions, velocities, torques, ground_truth_labels = self.load_data_at_ros_seq(
            self.first_index + idx)

        # Save the labels and number of nodes
        data.y = torch.tensor(ground_truth_labels, dtype=torch.float)
        data.num_nodes = self.URDF.get_num_nodes()

        # Create the feature matrices
        number_nodes = self.URDF.get_num_of_each_node_type()
        base_x = torch.ones((number_nodes[0], 1), dtype=torch.float)
        joint_x = torch.ones((number_nodes[1], 3), dtype=torch.float)
        foot_x = torch.ones((number_nodes[2], 1), dtype=torch.float)

        # For each joint specified
        for urdf_node_name in self.nodes_for_attributes:

            # Find the index of this particular node
            node_index = self.urdf_name_to_index[urdf_node_name]

            # Get the msg array index
            msg_ind = self.urdf_to_dataset_index[urdf_node_name]

            # Add the features to x matrix
            joint_x[node_index, 0] = positions[msg_ind]
            joint_x[node_index, 1] = velocities[msg_ind]
            joint_x[node_index, 2] = torques[msg_ind]

        # Save the matrices into the HeteroData object
        data['base'].x = base_x  # [num_papers, num_features_paper]
        data['joint'].x = joint_x  # [num_authors, num_features_author]
        data['foot'].x = foot_x  # [num_institutions, num_features_institution]
        return data

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


class CerberusDataset(FlexibleDataset):
    """
    Dataset class that can be used to load any of the A1 or Go1 robot
    rosbag sequences from the Cerberus state estimation dataset
    and turn it into a GNN.

    Here is where more information can be found on the dataset:
    https://github.com/ShuoYangRobotics/Cerberus?tab=readme-ov-file
    """

    def __init__(self,
                 root: str,
                 urdf_path: Path,
                 ros_builtin_path: str,
                 urdf_to_desc_path: str,
                 data_format: str = 'gnn',
                 transform=None,
                 pre_transform=None,
                 pre_filter=None):
        """
        Constructor for Cerberus Dataset and child classes.
        """
        super().__init__(root, urdf_path, ros_builtin_path, urdf_to_desc_path,
                         data_format, transform, pre_transform, pre_filter)

        # Map ROS names to array positions
        self.ros_name_in_index = [
            'FL0', 'FL1', 'FL2', 'FR0', 'FR1', 'FR2', 'RL0', 'RL1', 'RL2',
            'RR0', 'RR1', 'RR2', 'FL_foot', 'FR_foot', 'RL_foot', 'RR_foot'
        ]

        # Map urdf names to ROS names
        # How do I know this is the correct mapping?
        # Line 347 of https://github.com/ShuoYangRobotics/Cerberus/blob/main/src/main.cpp
        self.urdf_to_ros_map = {
            'FL_hip_joint': 'FL0',
            'FL_thigh_joint': 'FL1',
            'FL_calf_joint': 'FL2',
            'FR_hip_joint': 'FR0',
            'FR_thigh_joint': 'FR1',
            'FR_calf_joint': 'FR2',
            'RL_hip_joint': 'RL0',
            'RL_thigh_joint': 'RL1',
            'RL_calf_joint': 'RL2',
            'RR_hip_joint': 'RR0',
            'RR_thigh_joint': 'RR1',
            'RR_calf_joint': 'RR2',
            'FL_foot_fixed': 'FL_foot',
            'FR_foot_fixed': 'FR_foot',
            'RL_foot_fixed': 'RL_foot',
            'RR_foot_fixed': 'RR_foot'
        }

        # Open up the urdf file and get the edge matrix
        self.edge_matrix = self.URDF.get_edge_index_matrix()
        self.edge_matrix_tensor = torch.tensor(self.edge_matrix,
                                               dtype=torch.long)

        # Define the names and indicies that contain ground truth labels
        self.ground_truth_urdf_names = [
            'FL_foot_fixed', 'FR_foot_fixed', 'RL_foot_fixed', 'RR_foot_fixed'
        ]

        self.ground_truth_ros_names = []
        for urdf_name in self.ground_truth_urdf_names:
            self.ground_truth_ros_names.append(self.urdf_to_ros_map[urdf_name])

        self.ground_truth_graph_indices = []
        for urdf_name in self.ground_truth_urdf_names:
            self.ground_truth_graph_indices.append(
                self.urdf_name_to_index[urdf_name])

        self.ground_truth_array_indices = []
        for ros_name in self.ground_truth_ros_names:
            self.ground_truth_array_indices.append(
                self.get_index_of_ros_name(ros_name))

        # Define the nodes that should recieve features
        self.nodes_for_attributes = [
            'FL_hip_joint', 'FL_thigh_joint', 'FL_calf_joint', 'FR_hip_joint',
            'FR_thigh_joint', 'FR_calf_joint', 'RL_hip_joint',
            'RL_thigh_joint', 'RL_calf_joint', 'RR_hip_joint',
            'RR_thigh_joint', 'RR_calf_joint'
        ]

    def get_google_drive_file_id(self):
        """
        Method for child classes to choose which sequence to load.
        """
        raise self.notImplementedError

    def get_joint_msg_topic(self):
        """
        Method for child classes to specify the proper message topic
        depending on the robot.
        """
        raise self.notImplementedError

    @property
    def raw_file_names(self):
        return ['data.bag']

    def download(self):
        download_file_from_google_drive(self.get_google_drive_file_id(),
                                        Path(self.root, 'raw'), "data.bag")

    def process(self):
        topic = self.get_joint_msg_topic()

        # Set up a reader to read the rosbag
        path_to_bag = os.path.join(self.root, 'raw', 'data.bag')
        self.reader = AnyReader([Path(path_to_bag)])
        self.reader.open()
        self.joint_gen = self.reader.messages(connections=[
            x for x in self.reader.connections if x.topic == topic
        ])

        # Iterate through every joint message and save important data in txt file
        length = 0
        first_seq = None
        for connection, _timestamp, rawdata in self.joint_gen:
            msg = self.reader.deserialize(rawdata, connection.msgtype)
            if first_seq is None:
                first_seq = msg.header.seq
            with open(
                    os.path.join(self.processed_dir,
                                 str(msg.header.seq) + ".txt"), "w") as f:
                arrays = [msg.position, msg.velocity, msg.effort]
                for array in arrays:
                    for val in array:
                        f.write(str(val) + " ")
                    f.write('\n')
            length += 1

        # Write a txt file to save the dataset length & and first sequence index
        with open(os.path.join(self.processed_dir, "info.txt"), "w") as f:
            f.write(str(length) + " " + str(first_seq))

            # TODO: Add a note saying which dataset this is, and add
            # a check to make sure we don't load an improper dataset.

    def get_index_of_ros_name(self, name: str):
        """
        Given the ROS joint name, return the index
        in the array that its values can be found.
        """

        for i in range(0, len(self.ros_name_in_index)):
            if name == self.ros_name_in_index[i]:
                return i
        raise IndexError("This ROS joint name doesn't exist.")

    def load_data_at_ros_seq(self, ros_seq: int):
        # Open the file with the proper index
        positions, velocities, efforts = [], [], []
        with open(os.path.join(self.processed_dir,
                               str(ros_seq) + ".txt"), 'r') as f:
            line = f.readline().split(" ")[:-1]
            for i in range(0, len(line)):
                positions.append(float(line[i]))
            line = f.readline().split(" ")[:-1]
            for i in range(0, len(line)):
                velocities.append(float(line[i]))
            line = f.readline().split(" ")[:-1]
            for i in range(0, len(line)):
                efforts.append(float(line[i]))

        # Get the ground truth force labels
        # TODO: Write these methods for these functions
        ground_truth_labels = []
        for val in self.ground_truth_array_indices:
            ground_truth_labels.append(efforts[val])

        return positions, velocities, efforts, ground_truth_labels

    def get_helper_gnn(self, idx):
        # Load the rosbag information
        positions, velocities, efforts, ground_truth_labels = self.load_data_at_ros_seq(
            self.first_index + idx)

        # Create a node feature matrix
        x = torch.ones((self.URDF.get_num_nodes(), 2), dtype=torch.float)

        # For each joint specified
        for urdf_node_name in self.nodes_for_attributes:

            # Find the index of this particular node
            node_index = self.urdf_name_to_index[urdf_node_name]

            # Get the msg array index
            msg_ind = self.get_index_of_ros_name(
                self.urdf_to_ros_map[urdf_node_name])

            # Add the features to x matrix
            x[node_index, 0] = positions[msg_ind]
            x[node_index, 1] = velocities[msg_ind]

        # Create the graph
        graph = Data(x=x,
                     edge_index=self.edge_matrix_tensor,
                     y=torch.tensor(ground_truth_labels, dtype=torch.float),
                     num_nodes=self.URDF.get_num_nodes())
        return graph

    def get_helper_mlp(self, idx):
        # Load the rosbag information
        positions, velocities, efforts, ground_truth_labels = self.load_data_at_ros_seq(
            self.first_index + idx)

        # Make the network inputs
        x = torch.tensor((positions[:12] + velocities[:12]), dtype=torch.float)

        # Create the ground truth lables
        y = torch.tensor(ground_truth_labels, dtype=torch.float)
        return x, y


class CerberusA1Dataset(CerberusDataset):
    """
    Child class of Cerberus Dataset, but parent class
    of datasets that use the UniTree A1 robot.
    """

    def get_expected_urdf_name(self):
        return "a1"

    def get_joint_msg_topic(self):
        return "/hardware_a1/joint_foot"


class CerberusStreetDataset(CerberusA1Dataset):
    """
    This class specifically loads the "street" sequence
    of the Cerberus state estimation dataset.
    """

    def get_google_drive_file_id(self):
        return "1rVQW3VPx9WwpJAh8vWKELD0eW9yI_8Vu"

    def get_start_and_end_seq_ids(self):
        return 1597, 264904


class CerberusTrackDataset(CerberusA1Dataset):
    """
    This class specifically loads the "track" sequence
    of the Cerberus state estimation dataset.
    """

    def get_google_drive_file_id(self):
        return "1t2Y2Lp757lmYGsuGqu2T0aVnyKgZlLSW"

    def get_start_and_end_seq_ids(self):
        return 2603, 283736


class CerberusGo1Dataset(CerberusDataset):
    """
    Child class of Cerberus Dataset, but parent class
    of datasets that use the UniTree Gp1 robot.
    """

    def get_expected_urdf_name(self):
        return "go1"

    def get_joint_msg_topic(self):
        return "/hardware_go1/joint_foot"


class CerberusCampusDataset(CerberusGo1Dataset):
    """
    This class specifically loads the "campus" sequence
    of the Cerberus state estimation dataset.
    """

    def get_google_drive_file_id(self):
        return "1UinyXOQiVSPG6n2weYvekvP9NVV8_CLe"

    def get_start_and_end_seq_ids(self):
        return 119, 174560


def visualize_graph(pytorch_graph: Data,
                    robot_graph: NormalRobotGraph,
                    fig_save_path: Path = None,
                    draw_edges: bool = False):
    """
    This helper method visualizes a Data graph object
    using networkx.
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
