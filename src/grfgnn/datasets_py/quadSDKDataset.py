import torch
from torch_geometric.data import Data, Dataset, HeteroData
from ..graphParser import NormalRobotGraph, HeterogeneousRobotGraph
from rosbags.highlevel import AnyReader
from pathlib import Path
from torchvision.datasets.utils import download_file_from_google_drive
import os
from .flexibleDataset import FlexibleDataset
import numpy as np

class QuadSDKDataset(FlexibleDataset):
    """
    Dataset class for the simulated data provided through
    Quad-SDK and the Gazebo simulator.
    """

    # ========================= DOWNLOADING ==========================
    def get_downloaded_dataset_file_name(self):
        return "data.bag"

    # ========================= PROCESSING ===========================
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

                # Add on the robot pose information
                arrays.append([joint_data.body.pose.position.x,
                               joint_data.body.pose.position.y,
                               joint_data.body.pose.position.z])
                arrays.append([joint_data.body.pose.orientation.x,
                               joint_data.body.pose.orientation.y,
                               joint_data.body.pose.orientation.z,
                               joint_data.body.pose.orientation.w])

                for array in arrays:
                    for val in array:
                        f.write(str(val) + " ")
                    f.write('\n')

            # Track how many entries we have
            dataset_entries += 1

        # Write a txt file to save the dataset length, first sequence index,
        # and the download id (for ensuring we have the right dataset later)
        with open(os.path.join(self.processed_dir, "info.txt"), "w") as f:
            f.write(str(dataset_entries) + " " + str(0) + " " + self.get_google_drive_file_id())

    # ============= DATA SORTING ORDER AND MAPPINGS ==================    
    
    def get_base_node_sorted_order(self) -> list[str]:
        return ['floating_base']

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

    # ===================== DATASET PROPERTIES =======================
    def get_expected_urdf_name(self):
        return "a1"
    
    # ======================== DATA LOADING ==========================
    def load_data_at_dataset_seq(self, seq_num: int):
        grfs, lin_acc, ang_vel, positions, velocities, torques, r_p, r_quat = [], [], [], [], [], [], [], []
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
            line = f.readline().split(" ")[:-1]
            for i in range(0, len(line)):
                r_p.append(float(line[i]))
            line = f.readline().split(" ")[:-1]
            for i in range(0, len(line)):
                r_quat.append(float(line[i]))

        # Extract the ground truth Z GRF
        z_grfs = []
        for val in range(0, 4):
            start_index = val * 3
            z_grfs.append(grfs[start_index + 2])

        # Convert them all to numpy arrays
        lin_acc = np.array(lin_acc)
        ang_vel = np.array(ang_vel)
        positions = np.array(positions)
        velocities = np.array(velocities)
        torques = np.array(torques)
        z_grfs = np.array(z_grfs)
        r_p = np.array(r_p)
        r_quat = np.array(r_quat)

        return lin_acc, ang_vel, positions, velocities, torques, None, None, z_grfs, r_p, r_quat

# ================================================================
# ===================== DATASET SEQUENCES ========================
# ================================================================

class QuadSDKDataset_A1Speed0_5(QuadSDKDataset):
    def get_google_drive_file_id(self):
        return "17tvm0bmipTpueehUNQ-hJ8w5arc79q0M"

class QuadSDKDataset_A1Speed1_0(QuadSDKDataset):
    def get_google_drive_file_id(self):
        return "1qSdm8Rm6UazwhzCV5DfMHF0AoyKNrthf"

class QuadSDKDataset_A1Speed1_5FlippedOver(QuadSDKDataset):
    def get_google_drive_file_id(self):
        return "1h5CN-IIJlLnMvWp0sk5Ho-hiJq2NMqCT"