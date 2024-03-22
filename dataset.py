import torch
import torch_geometric
from torch_geometric.data import Data
from urdfParser import RobotURDF
from csvParser import RobotCSV
import networkx
import numpy as np
import matplotlib.pyplot as plt
from rosbags.highlevel import AnyReader
from torch_geometric.data import Dataset
from pathlib import Path
from itertools import islice

class CerberusStreetDataset(Dataset):

    def __init__(self,
                 root=None,
                 transform=None,
                 pre_transform=None,
                 pre_filter=None):
        super().__init__(root, transform, pre_transform, pre_filter)

        # Setup the reader for reading rosbags
        self.reader = AnyReader([Path(root)])
        self.reader.open()
        self.imu_gen = self.reader.messages(connections=[
            x for x in self.reader.connections 
            if x.topic == '/hardware_a1/imu'
        ])
        self.joint_gen = self.reader.messages(connections=[
            x for x in self.reader.connections
            if x.topic == '/hardware_a1/joint_foot'
        ])

        # Get the first index of the dataset.
        for connection, timestamp, rawdata in self.imu_gen:
            msg = self.reader.deserialize(rawdata, connection.msgtype)
            self.first_index = msg.header.seq
            break

        # Calculate the length of the dataset
        self.length = sum(1 for _ in self.imu_gen)
        
        # Close the reader
        self.reader.close()

    def len(self):
        return self.length

    def get_val_from_generator(self, gen, idx, default=None):
        # Quickly index into the generator and extract the
        # value at the idx position
        return next(islice(gen, idx, None), default)

    def get(self, idx):
        # Find the joint_foot message with the right index
        self.reader.open()
        connection, timestamp, rawdata = self.get_val_from_generator(self.joint_gen, idx)
        msg = self.reader.deserialize(rawdata, connection.msgtype)
        print(msg)

        # Create the graph
        # Data(edge_index=edge_matrix_tensor,
        #              edge_attr=torch.tensor(edge_attrs_matrices[i],
        #                                     dtype=torch.float),
        #              y=torch.tensor(y_all_graphs[i], dtype=torch.float),
        #              num_nodes=HyQ_URDF.get_num_links()))
        
        # Close the reader
        self.reader.close()
        pass


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
        edge_matrix = HyQ_URDF.get_edge_matrix()
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
                     num_nodes=HyQ_URDF.get_num_links()))
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
        edge_matrix = HyQ_URDF.get_edge_matrix()

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
            edge_name_to_connection_dict = HyQ_URDF.get_edge_name_to_connection_dict(
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
    edge_matrix = HyQ_URDF.get_edge_matrix()

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
                                  labels=HyQ_URDF.get_link_num_to_name_dict(),
                                  verticalalignment='top',
                                  font_size=8)
    networkx.draw_networkx_edge_labels(nx_graph,
                                       pos=spring_layout,
                                       edge_labels=edge_labels,
                                       rotate=False,
                                       font_size=7)
    plt.show()


def main():
    cerb = CerberusStreetDataset(
        '/home/dlittleman/state-estimation-gnn/datasets/street.bag')
    print(cerb.get(263306))


if __name__ == "__main__":
    main()
