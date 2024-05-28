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
import itertools


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

    def load_data_at_dataset_seq(self, ros_seq: int):
        """
        This helper function opens the file named "ros_seq"
        and loads the position, velocity, and effort information.

        Parameters:
            ros_seq (int): The sequence number of the ros message
                whose data should be loaded.
        """
        raise self.notImplementedError

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


class QuadSDKDataset(FlexibleDataset):
    """
    Dataset class for the simulated data provided through
    Quad-SDK and the Gazebo simulator.
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
        super().__init__(root, urdf_path, ros_builtin_path, urdf_to_desc_path,
                         data_format, transform, pre_transform, pre_filter)

        # Map urdf names to array indexes
        # TODO: Double check this
        self.urdf_name_to_dataset_array_index = {
            'FR_hip_joint': 6,
            'FR_thigh_joint': 7,
            'FR_calf_joint': 8,
            'FL_hip_joint': 0,
            'FL_thigh_joint': 1,
            'FL_calf_joint': 2,
            'RR_hip_joint': 9,
            'RR_thigh_joint': 10,
            'RR_calf_joint': 11,
            'RL_hip_joint': 3,
            'RL_thigh_joint': 4,
            'RL_calf_joint': 5,
            'FR_foot_fixed': 2,
            'FL_foot_fixed': 0,
            'RR_foot_fixed': 3,
            'RL_foot_fixed': 1,
        }

        # Define the node names for the robot feet
        self.foot_urdf_names = [
            'FR_foot_fixed',
            'FL_foot_fixed',
            'RR_foot_fixed',
            'RL_foot_fixed',
        ]

        # Get the indices of the feet nodes in the robotGraph
        self.foot_node_indices = []
        for urdf_name in self.foot_urdf_names:
            self.foot_node_indices.append(
                self.urdf_name_to_graph_index[urdf_name])

        # Get the GT grf array indices that correspond to the feet nodes order
        self.gt_grf_array_indices = []
        for urdf_name in self.foot_urdf_names:
            self.gt_grf_array_indices.append(
                self.urdf_name_to_dataset_array_index[urdf_name])

        # Define the nodes that should recieve features
        self.joint_nodes_for_attributes = [
            'FL_hip_joint', 'FL_thigh_joint', 'FL_calf_joint', 'FR_hip_joint',
            'FR_thigh_joint', 'FR_calf_joint', 'RL_hip_joint',
            'RL_thigh_joint', 'RL_calf_joint', 'RR_hip_joint',
            'RR_thigh_joint', 'RR_calf_joint'
        ]
        self.base_nodes_for_attributes = ['floating_base']

    def process(self):
        # Set up a reader to read the rosbag
        path_to_bag = os.path.join(
            self.root, 'raw', 'data.bag')
        self.reader = AnyReader([Path(path_to_bag)])
        self.reader.open()
        generator = self.reader.messages(connections=[])

        # Iterate through the generators and write important data
        # to a file
        prev_imu_msg, prev_grf_msg = None, None
        dataset_entries = 0
        for connection, _timestamp, rawdata in generator:
            # Save the most recent IMU and GRF ground truth data
            if connection.topic == '/robot_1/state/imu':
                prev_imu_msg = (rawdata, connection.msgtype)
            elif connection.topic == '/robot_1/state/grfs':
                prev_grf_msg = (rawdata, connection.msgtype)

            # Create a dataset entry, with the most recent joint
            # info, and the most recent IMU and GRF Ground Truth
            elif connection.topic == '/robot_1/state/ground_truth' \
                and prev_imu_msg is not None \
                and prev_grf_msg is not NotImplemented:

                grf_data = self.reader.deserialize(prev_grf_msg[0],
                                                   prev_grf_msg[1])
                joint_data = self.reader.deserialize(rawdata,
                                                     connection.msgtype)
                imu_data = self.reader.deserialize(prev_imu_msg[0],
                                                   prev_imu_msg[1])
                
                with open(str(Path(self.processed_dir,
                                str(dataset_entries) + ".txt")), "w") as f:
                    arrays = []

                    # Add on the GRF data
                    grf_vec = grf_data.vectors
                    arrays.append([grf_vec[0].x, grf_vec[0].y, grf_vec[0].z,
                                grf_vec[1].x, grf_vec[1].y, grf_vec[1].z,
                                grf_vec[2].x, grf_vec[2].y, grf_vec[2].z,
                                grf_vec[3].x, grf_vec[3].y, grf_vec[3].z])

                    # Add on the IMU data
                    arrays.append([
                        imu_data.linear_acceleration.x,
                        imu_data.linear_acceleration.y,
                        imu_data.linear_acceleration.z
                    ])
                    arrays.append([
                        imu_data.angular_velocity.x, 
                        imu_data.angular_velocity.y,
                        imu_data.angular_velocity.z
                    ])

                    # Add on the joint data
                    arrays.append(joint_data.joints.position)
                    arrays.append(joint_data.joints.velocity)
                    arrays.append(joint_data.joints.effort)

                    for array in arrays:
                        for val in array:
                            f.write(str(val) + " ")
                        f.write('\n')

                # Track how many entries we have
                dataset_entries += 1

            else: # Ignore other messages in the rosbag
                continue

        # Write a txt file to save the dataset length & and first sequence index
        with open(os.path.join(self.processed_dir, "info.txt"), "w") as f:
            f.write(str(dataset_entries) + " " + str(0))

    def get_expected_urdf_name(self):
        return "a1"

    def get_start_and_end_seq_ids(self):
        return 0, 14973

    def load_data_at_dataset_seq(self, seq_num: int):
        """
        Loads data from the dataset at the provided sequence number.
        """
        grfs, lin_acc, ang_vel, positions, velocities, torques = [], [], [], [], [], []
        with open(os.path.join(self.processed_dir,
                               str(seq_num) + ".txt"), 'r') as f:
            line = f.readline().split(" ")[:-1]
            for i in range(0, len(line)):
                grfs.append(float(line[i]))
            line = f.readline().split(" ")[:-1]
            for i in range(0, len(line)):
                lin_acc.append(float(line[i]))
            line = f.readline().split(" ")[:-1]
            for i in range(0, len(line)):
                ang_vel.append(float(line[i]))
            line = f.readline().split(" ")[:-1]
            for i in range(0, len(line)):
                positions.append(float(line[i]))
            line = f.readline().split(" ")[:-1]
            for i in range(0, len(line)):
                velocities.append(float(line[i]))
            line = f.readline().split(" ")[:-1]
            for i in range(0, len(line)):
                torques.append(float(line[i]))

        # Extract the ground truth Z GRF
        z_grfs = []
        for val in range(0, 4):
            start_index = val * 3
            z_grfs.append(grfs[start_index + 2])

        return lin_acc, ang_vel, positions, velocities, torques, z_grfs
    
    def load_data_sorted(self, seq_num: int):
        """
        Loads data from the dataset at the provided sequence number.
        However, the positions, velocities, and torques are sorted so that
        they match the order found in self.joint_nodes_for_attributes.
        Additionally, the Z_GRF_labels are sorted so it matches the order
        found in self.foot_urdf_names.
        """
        lin_acc, ang_vel, positions, velocities, torques, z_grfs = self.load_data_at_dataset_seq(seq_num)

        # Sort the joint information
        positions_sorted, velocities_sorted, torques_sorted = [], [], []
        for urdf_node_name in self.joint_nodes_for_attributes:
            array_index = self.urdf_name_to_dataset_array_index[urdf_node_name]
            positions_sorted.append(positions[array_index])
            velocities_sorted.append(velocities[array_index])
            torques_sorted.append(torques[array_index])

        # Sort the Z_GRF_labels
        z_grfs_sorted = []
        for index in self.gt_grf_array_indices:
            z_grfs_sorted.append(z_grfs[index])

        return lin_acc, ang_vel, positions_sorted, velocities_sorted, torques_sorted, z_grfs_sorted

    def get_helper_mlp(self, idx):
        # Load the rosbag information
        lin_acc, ang_vel, positions, velocities, torques, ground_truth_labels = self.load_data_sorted(
            self.first_index + idx)

        # Make the network inputs
        x = torch.tensor((lin_acc + ang_vel + positions + velocities + torques), dtype=torch.float)

        # Create the ground truth lables
        y = torch.tensor(ground_truth_labels, dtype=torch.float)
        return x, y
    
    def get_helper_gnn(self, idx):
        # Load the rosbag information
        lin_acc, ang_vel, positions, velocities, torques, z_grfs = self.load_data_sorted(
            self.first_index + idx)

        # Create a note feature matrix
        x = torch.ones((self.robotGraph.get_num_nodes(), 3), dtype=torch.float)

        # For each joint specified
        for i, urdf_node_name in enumerate(self.joint_nodes_for_attributes):

            # Find the index of this particular node
            node_index = self.urdf_name_to_graph_index[urdf_node_name]

            # Add the features to x matrix
            x[node_index, 0] = positions[i]
            x[node_index, 1] = velocities[i]
            x[node_index, 2] = torques[i]

        # Create the graph
        self.edge_matrix = self.robotGraph.get_edge_index_matrix()
        self.edge_matrix_tensor = torch.tensor(self.edge_matrix,
                                               dtype=torch.long)
        graph = Data(x=x,
                     edge_index=self.edge_matrix_tensor,
                     y=torch.tensor(z_grfs, dtype=torch.float),
                     num_nodes=self.robotGraph.get_num_nodes())
        return graph

    def get_helper_heterogeneous_gnn(self, idx):
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
               'joint'].edge_attr = torch.tensor(bj_attr, dtype=torch.long)
        data['joint', 'connect',
             'base'].edge_attr = torch.tensor(jb_attr, dtype=torch.long)
        data['joint', 'connect',
             'joint'].edge_attr = torch.tensor(jj_attr, dtype=torch.long)
        data['foot', 'connect',
             'joint'].edge_attr = torch.tensor(fj_attr, dtype=torch.long)
        data['joint', 'connect',
             'foot'].edge_attr = torch.tensor(jf_attr, dtype=torch.long)

        # Load the rosbag information
        lin_acc, ang_vel, positions, velocities, torques, z_grfs = self.load_data_sorted(
            self.first_index + idx)

        # Save the labels and number of nodes
        data.y = torch.tensor(z_grfs, dtype=torch.float)
        data.num_nodes = self.robotGraph.get_num_nodes()

        # Create the feature matrices
        number_nodes = self.robotGraph.get_num_of_each_node_type()
        base_x = torch.ones((number_nodes[0], 6), dtype=torch.float)
        joint_x = torch.ones((number_nodes[1], 3), dtype=torch.float)
        foot_x = torch.ones((number_nodes[2], 1), dtype=torch.float)

        # For each joint specified
        for i, urdf_node_name in enumerate(self.joint_nodes_for_attributes):

            # Find the index of this particular node
            node_index = self.urdf_name_to_graph_index[urdf_node_name]

            # Add the features to x matrix
            joint_x[node_index, 0] = positions[i]
            joint_x[node_index, 1] = velocities[i]
            joint_x[node_index, 2] = torques[i]

        # For each base specified (should be 1)
        for urdf_node_name in self.base_nodes_for_attributes:

            # Find the index of this particular node
            node_index = self.urdf_name_to_graph_index[urdf_node_name]

            # Add the features to x matrix
            base_x[node_index, 0] = lin_acc[0]
            base_x[node_index, 1] = lin_acc[1]
            base_x[node_index, 2] = lin_acc[2]
            base_x[node_index, 3] = ang_vel[0]
            base_x[node_index, 4] = ang_vel[1]
            base_x[node_index, 5] = ang_vel[2]

        # Save the matrices into the HeteroData object
        data['base'].x = base_x 
        data['joint'].x = joint_x
        data['foot'].x = foot_x
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
