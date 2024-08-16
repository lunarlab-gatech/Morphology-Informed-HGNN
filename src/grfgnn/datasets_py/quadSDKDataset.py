from rosbags.highlevel import AnyReader
from pathlib import Path
from torchvision.datasets.utils import download_file_from_google_drive
import os
from .flexibleDataset import FlexibleDataset
import numpy as np
import scipy.io as sio

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
        # into a dictionary
        prev_grf_time, prev_joint_time, prev_imu_time = 0, 0, 0
        dataset_entries = 0

        # Create arrays for all of the data types
        timestamps = np.empty((0, 3), dtype=np.float64)
        imu_acc = np.empty((0, 3), dtype=np.float64)
        imu_omega = np.empty((0, 3), dtype=np.float64)
        q = np.empty((0, 12), dtype=np.float64)
        qd = np.empty((0, 12), dtype=np.float64) 
        tau = np.empty((0, 12), dtype=np.float64) 
        F = np.empty((0, 12), dtype=np.float64)
        r_p = np.empty((0, 3), dtype=np.float64) 
        r_o = np.empty((0, 4), dtype=np.float64)

        # For each message
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

            # Add on the timestamp info
            timestamps = np.concatenate((timestamps, np.array([[grf_time, joint_time, imu_time]], dtype=np.float64)), axis=0)

            # Add on the GRF data
            grf_vec = grf_data.vectors
            grf_array = np.array([[grf_vec[0].x, grf_vec[0].y, grf_vec[0].z,
                                    grf_vec[1].x, grf_vec[1].y, grf_vec[1].z,
                                    grf_vec[2].x, grf_vec[2].y, grf_vec[2].z,
                                    grf_vec[3].x, grf_vec[3].y, grf_vec[3].z]], dtype=np.float64)
            F = np.concatenate((F, grf_array), axis=0)

            # Add on the IMU data
            imu_acc = np.concatenate((imu_acc, np.array([[imu_data.linear_acceleration.x,
                imu_data.linear_acceleration.y,
                imu_data.linear_acceleration.z]], dtype=np.float64)), axis=0)
            imu_omega = np.concatenate((imu_omega, np.array([[imu_data.angular_velocity.x,
                imu_data.angular_velocity.y,
                imu_data.angular_velocity.z]], dtype=np.float64)), axis=0)

            # Add on the joint data
            q = np.concatenate((q, np.array([joint_data.joints.position], dtype=np.float64)), axis=0)
            qd = np.concatenate((qd, np.array([joint_data.joints.velocity], dtype=np.float64)), axis=0)
            tau = np.concatenate((tau, np.array([joint_data.joints.effort], dtype=np.float64)), axis=0)

            # Add on the robot pose information
            r_p = np.concatenate((r_p, np.array([[joint_data.body.pose.position.x,
                            joint_data.body.pose.position.y,
                            joint_data.body.pose.position.z]], dtype=np.float64)), axis=0)
            r_o = np.concatenate((r_o, np.array([[joint_data.body.pose.orientation.x,
                            joint_data.body.pose.orientation.y,
                            joint_data.body.pose.orientation.z,
                            joint_data.body.pose.orientation.w]], dtype=np.float64)), axis=0)

            # Track how many entries we have
            dataset_entries += 1

        # Create the dictionary with all of the data
        data_dict = {
            'timestamps': timestamps,
            'imu_acc': imu_acc,
            'imu_omega': imu_omega,
            'q': q,
            'qd': qd, 
            'tau': tau ,
            'F': F,
            'r_p': r_p,
            'r_o': r_o 
        }

        # Close the bag
        self.reader.close()

        # Save the mat file
        sio.savemat(os.path.join(self.processed_dir, "data.mat"), data_dict, do_compression=True)

        # Write a txt file to save the dataset length, first sequence index,
        # and the download id (for ensuring we have the right dataset later)
        with open(os.path.join(self.processed_dir, "info.txt"), "w") as f:
            f.write(str(dataset_entries) + " " + self.get_google_drive_file_id())

    # ============= DATA SORTING ORDER AND MAPPINGS ==================    
    def get_urdf_name_to_dataset_array_index(self):
        return {
            'floating_base': 0,
            
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
        # Outline the indices to extract the just the Z-GRFs
        z_indices = [2, 5, 8, 11]

        # Convert them all to numpy arrays
        lin_acc = np.array(self.mat_data['imu_acc'][seq_num:seq_num+self.history_length]).reshape(self.history_length, 3)
        ang_vel = np.array(self.mat_data['imu_omega'][seq_num:seq_num+self.history_length]).reshape(self.history_length, 3)
        j_p = np.array(self.mat_data['q'][seq_num:seq_num+self.history_length]).reshape(self.history_length, 12)
        j_v = np.array(self.mat_data['qd'][seq_num:seq_num+self.history_length]).reshape(self.history_length, 12)
        j_T = np.array(self.mat_data['tau'][seq_num:seq_num+self.history_length]).reshape(self.history_length, 12)
        z_grfs = np.squeeze(np.array(self.mat_data['F'][seq_num:seq_num+self.history_length]).reshape(self.history_length, 12)[-1,z_indices])
        r_p = np.array(self.mat_data['r_p'][seq_num:seq_num+self.history_length]).reshape(self.history_length, 3)
        r_quat = np.array(self.mat_data['r_o'][seq_num:seq_num+self.history_length]).reshape(self.history_length, 4)
        return lin_acc, ang_vel, j_p, j_v, j_T, None, None, z_grfs, r_p, r_quat
    
    def load_data_at_dataset_seq_with_timestamps(self, seq_num: int):
        """
        Same as load_data_at_dataset_seq(), but extra return argument 
        on the end

        NOTE: even if self.normalize is set, this WON'T return
        normalized data. Use load_data_sorted() for that.

        Parameters:
            seq_num (int): The sequence number of the txt file
                whose data should be loaded.

        Returns:
            Same arguments as load_data_at_dataset_seq() +
            timestamps (np.array) - Array containing the timestamps of the data. The rows
                correspond to the history length and the columns correspond to the specific
                timestamp per data. Column 0 contains the grf_timestamp (for the GRF labels),
                Column 1 contains the joint_timestamp (for the joint data, and additionally
                the robot pose information), and Column 2 contains the imu_timestamp (for the
                linear acceleration and angular velocity of the IMU). This value is in the 
                shape of (history_length, 3).
        """
        # Outline the indices to extract the just the Z-GRFs
        z_indices = [2, 5, 8, 11]

        # Convert them all to numpy arrays
        timestamps = np.array(self.mat_data['timestamps'][seq_num:seq_num+self.history_length]).reshape(self.history_length, 3)
        lin_acc = np.array(self.mat_data['imu_acc'][seq_num:seq_num+self.history_length]).reshape(self.history_length, 3)
        ang_vel = np.array(self.mat_data['imu_omega'][seq_num:seq_num+self.history_length]).reshape(self.history_length, 3)
        j_p = np.array(self.mat_data['q'][seq_num:seq_num+self.history_length]).reshape(self.history_length, 12)
        j_v = np.array(self.mat_data['qd'][seq_num:seq_num+self.history_length]).reshape(self.history_length, 12)
        j_T = np.array(self.mat_data['tau'][seq_num:seq_num+self.history_length]).reshape(self.history_length, 12)
        z_grfs = np.squeeze(np.array(self.mat_data['F'][seq_num:seq_num+self.history_length]).reshape(self.history_length, 12)[-1,z_indices])
        r_p = np.array(self.mat_data['r_p'][seq_num:seq_num+self.history_length]).reshape(self.history_length, 3)
        r_quat = np.array(self.mat_data['r_o'][seq_num:seq_num+self.history_length]).reshape(self.history_length, 4)
        return lin_acc, ang_vel, j_p, j_v, j_T, None, None, z_grfs, r_p, r_quat, timestamps
    
    def load_data_sorted_with_timestamps(self, seq_num: int):
        """
        Same as load_data_sorted(), but also returns timestamps.

        Returns:
            Same values as load_data_at_dataset_seq_with_timestamps(), but order of
            values inside arrays have been sorted (and potentially
            normalized).
        """
        lin_acc, ang_vel, j_p, j_v, j_T, f_p, f_v, labels, r_p, r_o, timestamps = self.load_data_at_dataset_seq_with_timestamps(seq_num)

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
                    norm_arrs[i] = np.nan_to_num((array-np.mean(array,axis=0))/np.std(array, axis=0), copy=False, nan=0.0)

            return norm_arrs[0], norm_arrs[1], norm_arrs[2], norm_arrs[3], norm_arrs[4], norm_arrs[5], norm_arrs[6], labels_sorted, norm_arrs[7], norm_arrs[8], timestamps
        else:
            return lin_acc, ang_vel, sorted_list[0], sorted_list[1], sorted_list[2], sorted_foot_list[0], sorted_foot_list[1], labels_sorted, r_p, r_o, timestamps

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