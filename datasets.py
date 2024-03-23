import torch
import torch_geometric
from torch_geometric.data import Data, Dataset
from urdfParser import RobotURDF
from csvParser import RobotCSV
import networkx
import numpy as np
import matplotlib.pyplot as plt
from rosbags.highlevel import AnyReader
from pathlib import Path
from itertools import islice
import os
from rosbags.rosbag2 import Writer
from rosbags.typesys import Stores, get_typestore, get_types_from_msg
from rosbags.interfaces import ConnectionExtRosbag2
from typing import cast


class CerberusStreetDataset(Dataset):

    def __init__(self,
                 root,
                 robotURDF: RobotURDF,
                 entries_to_use=None,
                 transform=None,
                 pre_transform=None,
                 pre_filter=None):

        # Setup the directories for raw and processed data
        self.root = root
        super().__init__(root, transform, pre_transform, pre_filter)

        # Setup the reader for reading rosbags
        path_to_bag = os.path.join(root, 'raw', 'street.bag')
        reader = AnyReader([Path(path_to_bag)])
        reader.open()
        imu_gen = reader.messages(connections=[
            x for x in reader.connections if x.topic == '/hardware_a1/imu'
        ])

        # Get the first index of the dataset.
        for connection, timestamp, rawdata in imu_gen:
            msg = reader.deserialize(rawdata, connection.msgtype)
            self.first_index = msg.header.seq
            break

        # Calculate the length of the dataset
        self.length = sum(1 for _ in imu_gen)
        if entries_to_use is not None and entries_to_use <= self.length:
            self.length = entries_to_use
        reader.close()

        # Open up the A1 urdf file and get the edge matrix
        self.A1_URDF = robotURDF
        self.edge_matrix = self.A1_URDF.get_edge_index_matrix()
        self.edge_matrix_tensor = torch.tensor(self.edge_matrix,
                                               dtype=torch.long)
        # Get node name to index mapping
        self.urdf_name_to_index = self.A1_URDF.get_node_name_to_index_dict()

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

        # Define the names that contain ground truth labels
        self.ground_truth_urdf_names = [
            'FL_foot_fixed', 'FR_foot_fixed', 'RL_foot_fixed', 'RR_foot_fixed'
        ]
        self.ground_truth_ros_names = []
        for urdf_name in self.ground_truth_urdf_names:
            self.ground_truth_ros_names.append(self.urdf_to_ros_map[urdf_name])

        # Define the nodes that should recieve features
        self.nodes_for_attributes = [
            'FL_hip_joint', 'FL_thigh_joint', 'FL_calf_joint', 'FR_hip_joint',
            'FR_thigh_joint', 'FR_calf_joint', 'RL_hip_joint',
            'RL_thigh_joint', 'RL_calf_joint', 'RR_hip_joint',
            'RR_thigh_joint', 'RR_calf_joint'
        ]

    @property
    def raw_file_names(self):
        return ['street.bag']

    def download(self):
        raise Exception(
            "Please download the bag file from https://drive.google.com/drive/folders/1PdCLMVRqS97Tc9VbJY12EUl9UHpVoO_X and put it in <dataset_dir>/raw/"
        )

    @property
    def processed_file_names(self):
        return ['processed.txt']

    def process(self):
        topic = '/hardware_a1/joint_foot'

        # Set up a reader to read the rosbag
        path_to_bag = os.path.join(self.root, 'raw', 'street.bag')
        self.reader = AnyReader([Path(path_to_bag)])
        self.reader.open()
        self.joint_gen = self.reader.messages(connections=[
            x for x in self.reader.connections if x.topic == topic
        ])

        # Iterate through every joint message and save important data in txt file
        for connection, _timestamp, rawdata in self.joint_gen:
            msg = self.reader.deserialize(rawdata, connection.msgtype)

            with open(
                    os.path.join(self.processed_dir,
                                 str(msg.header.seq) + ".txt"), "w") as f:
                arrays = [msg.position, msg.velocity, msg.effort]
                for array in arrays:
                    for val in array:
                        f.write(str(val) + " ")
                    f.write('\n')

        # Write a txt file so we know we've finisehd
        with open(os.path.join(self.processed_dir, "processed.txt"), "w") as f:
            f.write(
                "Bag files have been successfully written to this directory.")

    def len(self):
        return self.length

    def get_position_of_ros_name(self, name):
        for i in range(0, len(self.ros_name_in_index)):
            if name == self.ros_name_in_index[i]:
                return i

    def get(self, idx):
        # Open the file with the proper index
        positions = []
        velocities = []
        efforts = []
        with open(
                os.path.join(self.processed_dir,
                             str(self.first_index + idx) + ".txt"), 'r') as f:
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
        ground_truth_labels = []
        for name in self.ground_truth_ros_names:
            ground_truth_labels.append(
                efforts[self.get_position_of_ros_name(name)])

        # Create a note feature matrix
        x = torch.ones((self.A1_URDF.get_num_nodes(), 2), dtype=torch.float)

        # For each joint specified
        for urdf_node_name in self.nodes_for_attributes:

            # Find the index of this particular node
            node_index = self.urdf_name_to_index[urdf_node_name]

            # Get the msg array index
            msg_ind = self.get_position_of_ros_name(
                self.urdf_to_ros_map[urdf_node_name])

            # Add the features to x matrix
            x[node_index, 0] = positions[msg_ind]
            x[node_index, 1] = velocities[msg_ind]

        # Create the graph
        graph = Data(x=x,
                     edge_index=self.edge_matrix_tensor,
                     y=torch.tensor(ground_truth_labels, dtype=torch.float),
                     num_nodes=self.A1_URDF.get_num_nodes())

        return graph


class HyQDataset(Dataset):

    def __init__(self,
                 root=None,
                 transform=None,
                 pre_transform=None,
                 pre_filter=None):
        super().__init__(root, transform, pre_transform, pre_filter)
        print("Loading HyQ Dataset: This may take awhile...")

        # Load the HyQ URDF file & dataset
        HyQ_URDF = RobotURDF('urdf_files/HyQ/hyq.urdf',
                             'package://hyq_description/', 'hyq-description')
        HyQ_CSV = RobotCSV("trot_in_lab_1")

        # Extract the edge matrix and convert to tensor
        edge_matrix = HyQ_URDF.get_edge_index_matrix()
        edge_matrix_tensor = torch.tensor(edge_matrix, dtype=torch.long)

        # Create edge attribute matrices
        joint_to_features_dict = {
            # LF features
            'lf_haa_joint': ['LF_HAA_q', 'LF_HAA_qd_f'],
            'lf_hfe_joint': ['LF_HFE_q', 'LF_HFE_qd_f'],
            'lf_kfe_joint': ['LF_KFE_q', 'LF_KFE_qd_f'],
            # LH features
            'lh_haa_joint': ['LH_HAA_q', 'LH_HAA_qd_f'],
            'lh_hfe_joint': ['LH_HFE_q', 'LH_HFE_qd_f'],
            'lh_kfe_joint': ['LH_KFE_q', 'LH_KFE_qd_f'],
            # RF features
            'rf_haa_joint': ['RF_HAA_q', 'RF_HAA_qd_f'],
            'rf_hfe_joint': ['RF_HFE_q', 'RF_HFE_qd_f'],
            'rf_kfe_joint': ['RF_KFE_q', 'RF_KFE_qd_f'],
            # RH features
            'rh_haa_joint': ['RH_HAA_q', 'RH_HAA_qd_f'],
            'rh_hfe_joint': ['RH_HFE_q', 'RH_HFE_qd_f'],
            'rh_kfe_joint': ['RH_KFE_q', 'RH_KFE_qd_f'],
        }
        features_per_joint = 2
        edge_attrs_matrices = self._create_edge_attr_matrices(
            HyQ_CSV, HyQ_URDF, joint_to_features_dict, features_per_joint)

        # Create target matrix
        target_matrix_names = [
            'LF_HAA_fm', 'LF_HFE_fm', 'LF_KFE_fm', 'LH_HAA_fm', 'LH_HFE_fm',
            'LH_KFE_fm', 'RF_HAA_fm', 'RF_HFE_fm', 'RF_KFE_fm', 'RH_HAA_fm',
            'RH_HFE_fm', 'RH_KFE_fm'
        ]
        y_all_graphs = self._create_target_matrix(HyQ_CSV, HyQ_URDF,
                                                  target_matrix_names)

        # Create a list of Data (graph) objects
        self.dataset = list()
        for i in range(0, len(edge_attrs_matrices)):
            self.dataset.append(
                Data(edge_index=edge_matrix_tensor,
                     edge_attr=torch.tensor(edge_attrs_matrices[i],
                                            dtype=torch.float),
                     y=torch.tensor(y_all_graphs[i], dtype=torch.float),
                     num_nodes=HyQ_URDF.get_num_nodes()))
        print("HyQ data load complete")

    def _find_edge_indexes_from_connections(self, edge_matrix,
                                            edge_connections):
        edge_index = list()
        for i in range(0, len(edge_matrix[0])):
            vector = edge_matrix[:, i]
            if (np.array_equal(vector, edge_connections)
                    or np.array_equal(vector, np.flip(edge_connections))):
                edge_index.append(i)
        return edge_index

    def _create_edge_attr_matrices(self, HyQ_CSV: RobotCSV,
                                   HyQ_URDF: RobotURDF,
                                   joint_to_features_dict: dict[str,
                                                                list[str]],
                                   features_per_joint):
        # Load the edge matrix
        edge_matrix = HyQ_URDF.get_edge_index_matrix()

        # Create the edge_attribute matrix
        edge_attrs = np.zeros((HyQ_CSV.num_dataset_entries(),
                               len(edge_matrix[0]), features_per_joint))

        # For each joint specified
        for joint in joint_to_features_dict:
            # Get the desired feature names
            feature_names = joint_to_features_dict[joint]

            # Create the feature vector
            feature_vectors = np.zeros(
                (HyQ_CSV.num_dataset_entries(), len(feature_names)))
            for i in range(0, len(feature_names)):
                feature_vectors[:, i] = HyQ_CSV.pull_values(feature_names[i])

            # Find the indexes of the particular joint
            edge_name_to_connection_dict = HyQ_URDF.get_edge_name_to_connections_dict(
            )
            joint_edges_dict = edge_name_to_connection_dict[joint]
            joint_edge_indexes = self._find_edge_indexes_from_connections(
                edge_matrix, joint_edges_dict)

            # Add the feature vectors to the specified indexes
            for entry_index in range(0, HyQ_CSV.num_dataset_entries()):
                for edge_index in joint_edge_indexes:
                    edge_attrs[entry_index,
                               edge_index, :] = feature_vectors[entry_index, :]

        return edge_attrs

    def _create_target_matrix(self, HyQ_CSV: RobotCSV, HyQ_URDF: RobotURDF,
                              target_matrix_names: list[str]):
        # Create the target matrix
        y = np.zeros((HyQ_CSV.num_dataset_entries(), len(target_matrix_names)))

        # For each target name
        for i in range(0, len(target_matrix_names)):
            # Pull the desired values for this target
            target_values = HyQ_CSV.pull_values(target_matrix_names[i])

            # Save it in the target matrix
            y[:, i] = target_values

        return y

    def len(self):
        return len(self.dataset)

    def get(self, idx):
        return self.dataset[idx]


def hyq_test():
    # Load the HyQ URDF file & dataset
    HyQ_URDF = RobotURDF('urdf_files/HyQ/hyq.urdf')

    # Extract the edge matrix and convert to tensor
    edge_matrix = HyQ_URDF.get_edge_index_matrix()

    # Create the dataset
    dataset = HyQDataset()

    # Extract the first graph
    graph: Data = dataset[0]

    # Write the edge attributes onto the labels
    edge_labels = HyQ_URDF.get_edge_connection_to_name_dict()
    for i in range(0, len(graph.edge_attr)):
        edge_features = graph.edge_attr[i].numpy()

        # Only write features if they aren't all zeros
        if (np.sum(edge_features) != 0):
            # Find connection numbers for this edge
            connection_matrix = graph.edge_index[:, i].numpy()
            connection_tuple = (connection_matrix[0], connection_matrix[1])

            # Update the edge label
            try:
                label = edge_labels[connection_tuple]
                label += ": " + str(edge_features)
                edge_labels[connection_tuple] = label
            except KeyError:
                pass

    # Convert to networkx graph
    nx_graph = torch_geometric.utils.to_networkx(graph, to_undirected=True)

    # Draw the graph
    spring_layout = networkx.spring_layout(nx_graph)
    networkx.draw(nx_graph, pos=spring_layout)
    networkx.draw_networkx_labels(nx_graph,
                                  pos=spring_layout,
                                  labels=HyQ_URDF.get_node_num_to_name_dict(),
                                  verticalalignment='top',
                                  font_size=8)
    networkx.draw_networkx_edge_labels(nx_graph,
                                       pos=spring_layout,
                                       edge_labels=edge_labels,
                                       rotate=False,
                                       font_size=7)
    plt.show()


def a1_test():
    # Load the A1 URDF file
    A1_URDF = RobotURDF('urdf_files/A1/a1.urdf', 'package://a1_description/',
                        'unitree_ros/robots/a1_description', True)

    # Create the dataset
    dataset = CerberusStreetDataset(
        '/home/dlittleman/state-estimation-gnn/datasets/cerberus_street',
        A1_URDF)

    # Extract the first graph
    graph: Data = dataset[0]

    # Write the features onto the names
    node_labels = A1_URDF.get_node_index_to_name_dict()
    for i in range(0, len(graph.x)):
        label = node_labels[i]
        label += ": " + str(graph.x[i].numpy())
        node_labels[i] = label

    # Convert to networkx graph
    nx_graph = torch_geometric.utils.to_networkx(graph, to_undirected=True)

    # Draw the graph
    spring_layout = networkx.spring_layout(nx_graph)
    networkx.draw(nx_graph, pos=spring_layout)
    networkx.draw_networkx_labels(nx_graph,
                                  pos=spring_layout,
                                  labels=node_labels,
                                  verticalalignment='top',
                                  font_size=8)
    # networkx.draw_networkx_edge_labels(
    #     nx_graph,
    #     pos=spring_layout,
    #     edge_labels=A1_URDF.get_edge_connections_to_name_dict(),
    #     rotate=False,
    #     font_size=7)
    plt.show()


def main():
    a1_test()


if __name__ == "__main__":
    main()
