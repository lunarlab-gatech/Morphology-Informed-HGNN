import torch
from torch_geometric.data import Data, Dataset, HeteroData
from ..graphParser import NormalRobotGraph, HeterogeneousRobotGraph
from rosbags.highlevel import AnyReader
from pathlib import Path
from torchvision.datasets.utils import download_file_from_google_drive
import os
from .flexibleDataset import FlexibleDataset

class QuadSDKDataset(FlexibleDataset):
    """
    Dataset class for the simulated data provided through
    Quad-SDK and the Gazebo simulator.
    """

    def get_urdf_name_to_dataset_array_index(self):
        return {
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
            'RL_foot_fixed': 1 }

    def get_joint_node_sorted_order(self) -> list[str]:
        return [ 'FL_hip_joint', 'FL_thigh_joint', 'FL_calf_joint',
                 'FR_hip_joint', 'FR_thigh_joint', 'FR_calf_joint',
                 'RL_hip_joint', 'RL_thigh_joint', 'RL_calf_joint',
                 'RR_hip_joint', 'RR_thigh_joint', 'RR_calf_joint']

    def get_foot_node_sorted_order(self) -> list[str]:
        return ['FR_foot_fixed',
                'FL_foot_fixed',
                'RR_foot_fixed',
                'RL_foot_fixed']

    def process(self):
        # Set up a reader to read the rosbag
        path_to_bag = os.path.join(
            self.root, 'raw', 'data.bag')
        self.reader = AnyReader([Path(path_to_bag)])
        self.reader.open()
        connections = [x for x in self.reader.connections if x.topic == '/quadruped_dataset_entries']

        # Iterate through the generators and write important data
        # to a file
        prev_grf_time, prev_joint_time, prev_imu_time = 0, 0, 0
        dataset_entries = 0
        for connection, _timestamp, rawdata in self.reader.messages(connections=connections):

            data = self.reader.deserialize(rawdata, connection.msgtype)
            grf_data = data.grfs
            joint_data = data.joints
            imu_data = data.imu

            # Ensure that the messages are in time order
            # If they are, then we won't throw an error, so we can
            # guarantee then are in order if it works

            # We do assume that if two messages have the same exact timestamp, the one
            # that came after is after time-wise
            grf_time = grf_data.header.stamp.sec + (grf_data.header.stamp.nanosec / 1e9)
            joint_time = joint_data.header.stamp.sec + (joint_data.header.stamp.nanosec / 1e9)
            imu_time = imu_data.header.stamp.sec + (imu_data.header.stamp.nanosec / 1e9)

            if prev_grf_time > grf_time or prev_joint_time > joint_time or prev_imu_time > imu_time:
                raise ValueError("Rosbag entries aren't in timestamp order.")

            prev_grf_time = grf_time
            prev_joint_time = joint_time
            prev_imu_time = imu_time

            # Save the important data
            with open(str(Path(self.processed_dir,
                            str(dataset_entries) + ".txt")), "w") as f:
                arrays = []

                # Add on the timestamp info
                arrays.append([grf_time, joint_time, imu_time])

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

        # Write a txt file to save the dataset length & and first sequence index
        with open(os.path.join(self.processed_dir, "info.txt"), "w") as f:
            f.write(str(dataset_entries) + " " + str(0))

    def get_expected_urdf_name(self):
        return "a1"
    
    def get_downloaded_dataset_file_name(self):
        return "data.bag"

    def load_data_at_dataset_seq(self, seq_num: int):
        grfs, lin_acc, ang_vel, positions, velocities, torques = [], [], [], [], [], []
        with open(os.path.join(self.processed_dir,
                               str(seq_num) + ".txt"), 'r') as f:

            # Skip the timestamp info
            line = f.readline().split(" ")[:-1]

            # Start reading data
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

        return lin_acc, ang_vel, positions, velocities, torques, None, None, z_grfs

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
        lin_acc, ang_vel, positions, velocities, torques, z_grfs, ang_acc, joint_acc = self.load_data_sorted(
            self.first_index + idx + self.history_length - 1)
        data.y = torch.tensor(z_grfs, dtype=torch.float64)
        data.num_nodes = self.robotGraph.get_num_nodes()

        # Create the feature matrices
        number_nodes = self.robotGraph.get_num_of_each_node_type()
        base_x = torch.ones((number_nodes[0], 9 * self.history_length), dtype=torch.float64)
        joint_x = torch.ones((number_nodes[1], 4 * self.history_length), dtype=torch.float64)
        foot_x = torch.ones((number_nodes[2], 1), dtype=torch.float64)

        # For each dataset entry we include in the history
        for j in range(0, self.history_length):
            # Load the data for this entry
            lin_acc, ang_vel, positions, velocities, torques, z_grfs, ang_acc, joint_acc = self.load_data_sorted(self.first_index + idx + j)

            # For each joint specified
            for i, urdf_node_name in enumerate(self.joint_nodes_for_attributes):

                # Find the index of this particular node
                node_index = self.urdf_name_to_graph_index[urdf_node_name]

                # Add the features to x matrix
                joint_x[node_index, 0*self.history_length+j] = positions[i]
                joint_x[node_index, 1*self.history_length+j] = velocities[i]
                joint_x[node_index, 2*self.history_length+j] = joint_acc[i]
                joint_x[node_index, 3*self.history_length+j] = torques[i]

            # For each base specified (should be 1)
            for urdf_node_name in self.base_nodes_for_attributes:

                # Find the index of this particular node
                node_index = self.urdf_name_to_graph_index[urdf_node_name]

                # Add the features to x matrix
                base_x[node_index, 0*self.history_length+j] = lin_acc[0]
                base_x[node_index, 1*self.history_length+j] = lin_acc[1]
                base_x[node_index, 2*self.history_length+j] = lin_acc[2]
                base_x[node_index, 3*self.history_length+j] = ang_vel[0]
                base_x[node_index, 4*self.history_length+j] = ang_vel[1]
                base_x[node_index, 5*self.history_length+j] = ang_vel[2]
                base_x[node_index, 6*self.history_length+j] = ang_acc[0]
                base_x[node_index, 7*self.history_length+j] = ang_acc[1]
                base_x[node_index, 8*self.history_length+j] = ang_acc[2]

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

class QuadSDKDataset_A1Speed0_5(QuadSDKDataset):
    def get_google_drive_file_id(self):
        return "17tvm0bmipTpueehUNQ-hJ8w5arc79q0M"
    
    def get_start_and_end_seq_ids(self):
        return 0, 14882

class QuadSDKDataset_A1Speed1_0(QuadSDKDataset):
    def get_google_drive_file_id(self):
        return "1qSdm8Rm6UazwhzCV5DfMHF0AoyKNrthf"
    
    def get_start_and_end_seq_ids(self):
        return 0, 17530

class QuadSDKDataset_A1Speed1_5FlippedOver(QuadSDKDataset):
    def get_google_drive_file_id(self):
        return "1h5CN-IIJlLnMvWp0sk5Ho-hiJq2NMqCT"
    
    def get_start_and_end_seq_ids(self):
        return 0, 15692
