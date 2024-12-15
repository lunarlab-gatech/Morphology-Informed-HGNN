from rosbags.highlevel import AnyReader
from pathlib import Path
from torchvision.datasets.utils import download_file_from_google_drive
import os
from .flexibleDataset import FlexibleDataset
import numpy as np
import scipy.io as sio
from scipy.spatial.transform import Rotation

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
            grf_array = np.array([[ grf_vec[0].x, grf_vec[0].y, grf_vec[0].z,
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
            file_id, loc = self.get_file_id_and_loc()
            f.write(str(dataset_entries) + " " + file_id)
    
# ================================================================
# ======================== SUBCLASSES ============================
# ================================================================

class QuadSDKDataset_A1_DEPRECATED(QuadSDKDataset):
    """
    Note: This class is deprecated, but is kept around to support 
    test cases that ensure the FlexibleDataset is working. Don't
    use this class for any development, as the units/frames of the
    values aren't verified and the sorting values might not be 
    correct.
    """

    # ===================== DATASET PROPERTIES =======================
    def get_expected_urdf_name(self):
        return "a1"
    
    # ============= DATA SORTING ORDER AND MAPPINGS ==================    
    def get_urdf_name_to_dataset_array_index(self):
        # Our URDF order for A1 robot is FR, FL, RR, RL.
        
        return {
            # Order of joint data can be found here: https://github.com/robomechanics/quad-sdk/wiki/FAQ.
            # Note: Because they refer to the first joint as the abduction/adduction joint, and the second 
            # joint as the hip joint, this means that our "hip_joint" corresponds to their abduction/adduction 
            # joint, and our "thigh_joint" corresponds to what they call the hip joint.
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

            # Label order is from here: https://github.com/lunarlab-gatech/quad_sdk_fork/blob/devel/quad_simulator/gazebo_scripts/src/contact_state_publisher.cpp,
            # referring to the URDF to see order of toes. It's the same order as the joint data.

            'FR_foot_fixed': 2,
            'FL_foot_fixed': 0,
            'RR_foot_fixed': 3,
            'RL_foot_fixed': 1 
        }
    
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
        timestamps = np.array(self.mat_data['timestamps'][seq_num:seq_num+self.history_length]).reshape(self.history_length, 3)

        return lin_acc, ang_vel, j_p, j_v, j_T, None, None, z_grfs, r_p, r_quat, timestamps

class QuadSDKDataset_A1(QuadSDKDataset):
    # ===================== DATASET PROPERTIES =======================
    def get_expected_urdf_name(self):
        return "a1_description"
    
    # ============= DATA SORTING ORDER AND MAPPINGS ==================    
    def get_urdf_name_to_dataset_array_index(self):
        # Our URDF order for the Go2 is FR, FL, RR, RL.

        return {
            'floating_base': 0,
            
            # Joint Order is specified here: https://github.com/robomechanics/quad-sdk/wiki/FAQ .
            # It goes abd->hip->knee, which in our framing, is hip->thigh->calf, and this matches our URDF order.
            # Leg order goes FL, RL, FR, RR, which matches our URDF order of FL, RL, FR, RR.
            # Further proof for this can be found in quad_sdk_fork/quad_simulator/gazebo_scripts/src/estimator_plugin.cpp.

            # FL Leg
            '8': 0,
            '0': 1,
            '1': 2,

            # RL Leg
            '9': 3,
            '2': 4,
            '3': 5,

            # FR Leg
            '10': 6,
            '4': 7,
            '5': 8,

            # RR Leg
            '11': 9,
            '6': 10,
            '7': 11,

            # Feet orders (which matches leg orders)
            # Manually verified by reviewing scripts in quad_sdk_fork/quad_simulator/gazebo_scripts/src.
            'jtoe0': 0,
            'jtoe1': 1,
            'jtoe2': 2,
            'jtoe3': 3,
        }

    
    def urdf_to_pin_order_mapping(self):
        """
        See flexibleDataset.py for definition of this function.

        Nb joints = 14 (nq=19,nv=18)
            Joint 0 universe: parent=0
            Joint 1 root_joint: parent=0
            Joint 2 10: parent=1
            Joint 3 4: parent=2
            Joint 4 5: parent=3
            Joint 5 11: parent=1
            Joint 6 6: parent=5
            Joint 7 7: parent=6
            Joint 8 8: parent=1
            Joint 9 0: parent=8
            Joint 10 1: parent=9
            Joint 11 9: parent=1
            Joint 12 2: parent=11
            Joint 13 3: parent=12

             Therefore pinnochio order is FR, RR, FL, RL.
           The A1-Quad URDF order is FL, RL, FR, RR.
           This gives the URDF->Pin mapping seen below.
        """

        # Specifically for the Go2 Robot. 
        return [6, 7, 8, 9, 10, 11, 0, 1, 2, 3, 4, 5], [2, 3, 0, 1]
    
    def pin_to_urdf_order_mapping(self):
        return [6, 7, 8, 9, 10, 11, 0, 1, 2, 3, 4, 5], [2, 3, 0, 1]

    # ======================== DATA LOADING ==========================
    def load_data_at_dataset_seq(self, seq_num: int):
        """
        The units of the return values are as follows:
        - lin_acc (meters/sec^2, represented in the world frame (converted from body frame))
        - ang_vel (rad/sec, represented in the world frame (converted from body frame))
        - j_p (rad)
        - j_v (rad/sec)
        - j_T (Newton-meters)
        - z_grfs (Newtons, represented in the world frame)
        - r_p (meters, represented in the world frame)
        - r_quat (N/A (Quaternion), represented in the world frame) (x, y, z, w)
        - timestamps (secs)

        This dataset does have the following values available, but doesn't load
        them because they aren't employed by the Dynamics model, so we don't
        want them included in the learning model automatically.

        - f_p (meters, represented in robot's body frame) 
        - f_v (meters/sec, represented in robot's body frame)
        """

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
        timestamps = np.array(self.mat_data['timestamps'][seq_num:seq_num+self.history_length]).reshape(self.history_length, 3)

        # Convert the lin_acc and ang_vel from body frame to world frame
        for i in range(0, self.history_length):
            world_to_body_R = Rotation.from_quat(r_quat[i])
            body_to_world_R = world_to_body_R.inv().as_matrix()
            lin_acc_T = np.array([[lin_acc[i][0]], [lin_acc[i][1]], [lin_acc[i][2]]], dtype=np.double)
            ang_vel_T = np.array([[ang_vel[i][0]], [ang_vel[i][1]], [ang_vel[i][2]]], dtype=np.double)
            lin_acc[i] = ((body_to_world_R @ lin_acc_T).T)[0]
            ang_vel[i] = ((body_to_world_R @ ang_vel_T).T)[0]

        return lin_acc, ang_vel, j_p, j_v, j_T, None, None, z_grfs, r_p, r_quat, timestamps


class QuadSDKDataset_Go2(QuadSDKDataset):
    # ===================== DATASET PROPERTIES =======================
    def get_expected_urdf_name(self):
        return "go2_description"
    
    # ============= DATA SORTING ORDER AND MAPPINGS ==================    
    def get_urdf_name_to_dataset_array_index(self):
        # Our URDF order for the Go2 is FR, FL, RR, RL.

        return {
            'floating_base': 0,
            
            # Joint Order is specified here: https://github.com/robomechanics/quad-sdk/wiki/FAQ .
            # It goes abd->hip->knee, which in our framing, is hip->thigh->calf, and this matches
            # our URDF order.
            # Leg order goes FL, RL, FR, RR, which matches our URDF order of FL, RL, FR, RR.

            # FL Leg
            '8': 0,
            '0': 1,
            '1': 2,

            # RL Leg
            '9': 3,
            '2': 4,
            '3': 5,

            # FR Leg
            '10': 6,
            '4': 7,
            '5': 8,

            # RR Leg
            '11': 9,
            '6': 10,
            '7': 11,

            # Feet orders (which matches leg orders)
            'jtoe0': 0,
            'jtoe1': 1,
            'jtoe2': 2,
            'jtoe3': 3,
        }

    
    def urdf_to_pin_order_mapping(self):
        """
        See flexibleDataset.py for definition of this function.

        Nb joints = 14 (nq=19,nv=18)
            Joint 0 universe: parent=0
            Joint 1 root_joint: parent=0
            Joint 2 10: parent=1
            Joint 3 4: parent=2
            Joint 4 5: parent=3
            Joint 5 11: parent=1
            Joint 6 6: parent=5
            Joint 7 7: parent=6
            Joint 8 8: parent=1
            Joint 9 0: parent=8
            Joint 10 1: parent=9
            Joint 11 9: parent=1
            Joint 12 2: parent=11
            Joint 13 3: parent=12

             Therefore their order is FR, RR, FL, RL.
           The Go2-Quad URDF order is FL, RL, FR, RR.
           This gives the URDF->Pin mapping seen below.
        """

        # Specifically for the Go2 Robot. 
        return [6, 7, 8, 9, 10, 11, 0, 1, 2, 3, 4, 5], [2, 3, 0, 1]
    
    def pin_to_urdf_order_mapping(self):
        return [6, 7, 8, 9, 10, 11, 0, 1, 2, 3, 4, 5], [2, 3, 0, 1]

    # ======================== DATA LOADING ==========================
    def load_data_at_dataset_seq(self, seq_num: int):
        """
        The units of the return values are as follows:
        - lin_acc (meters/sec^2, represented in the world frame (converted from body frame))
        - ang_vel (rad/sec, represented in the world frame (converted from body frame))
        - j_p (rad)
        - j_v (rad/sec)
        - j_T (Newton-meters)
        - z_grfs (Newtons, represented in the world frame)
        - r_p (meters, represented in the world frame)
        - r_quat (N/A (Quaternion), represented in the world frame) (x, y, z, w)
        - timestamps (secs)

        This dataset does have the following values available, but doesn't load
        them because they aren't employed by the Dynamics model, so we don't
        want them included in the learning model automatically.

        - f_p (meters, represented in robot's body frame) 
        - f_v (meters/sec, represented in robot's body frame)
        """

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
        timestamps = np.array(self.mat_data['timestamps'][seq_num:seq_num+self.history_length]).reshape(self.history_length, 3)

        # Convert the lin_acc and ang_vel from body frame to world frame
        for i in range(0, self.history_length):
            world_to_body_R = Rotation.from_quat(r_quat[i])
            body_to_world_R = world_to_body_R.inv().as_matrix()
            lin_acc_T = np.array([[lin_acc[i][0]], [lin_acc[i][1]], [lin_acc[i][2]]], dtype=np.double)
            ang_vel_T = np.array([[ang_vel[i][0]], [ang_vel[i][1]], [ang_vel[i][2]]], dtype=np.double)
            lin_acc[i] = ((body_to_world_R @ lin_acc_T).T)[0]
            ang_vel[i] = ((body_to_world_R @ ang_vel_T).T)[0]

        return lin_acc, ang_vel, j_p, j_v, j_T, None, None, z_grfs, r_p, r_quat, timestamps

# ================================================================
# ===================== DATASET SEQUENCES ========================
# ================================================================

# A1_DEPRECATED
class QuadSDKDataset_A1Speed0_5_DEPRECATED(QuadSDKDataset_A1_DEPRECATED):
    def get_file_id_and_loc(self):
        return "https://www.dropbox.com/scl/fi/q8t67zc6yeyb78nqngqu2/QuadSDK-A1Speed0.5-OnlyForTestCases.bag?rlkey=38ihtx8mxkvh4cspxsm8xrudj&st=ftw2k75k&dl=1", "Dropbox"

class QuadSDKDataset_A1Speed1_0_DEPRECATED(QuadSDKDataset_A1_DEPRECATED):
    def get_file_id_and_loc(self):
        return "https://www.dropbox.com/scl/fi/8o8mkw5079yers8i0rhhs/QuadSDK-A1Speed1.0-OnlyForTestCases.bag?rlkey=mkigp3t3253py9ih5tskl3w8c&st=m7o3n6yg&dl=1", "Dropbox"

class QuadSDKDataset_A1Speed1_5FlippedOver_DEPRECATED(QuadSDKDataset_A1_DEPRECATED):
    def get_file_id_and_loc(self):
        return "https://www.dropbox.com/scl/fi/irxyiafwpn4rflrtfrb1g/QuadSDK-A1Speed1.5FlippedOver-OnlyForTestCases.bag?rlkey=0ldqimfn2jogwax6tdkq6vvqu&st=xrni7lp8&dl=1", "Dropbox"
    
# A1
class QuadSDKDataset_A1_Alpha(QuadSDKDataset_A1):
    # Speed: 0.5 mps, Terrain: Flat, Mu: 50
    def get_file_id_and_loc(self):
        return "https://www.dropbox.com/scl/fi/w2b7uk3sm3u42uxxnzm0o/robot_1_a1_0p5mps_flat_mu1_50_mu2_50_trial1_2024-09-09-14-22-23.bag?rlkey=435hyxui9wc2qt93sr1lq58az&st=yn0n3aol&dl=1", "Dropbox"

class QuadSDKDataset_A1_Bravo(QuadSDKDataset_A1):
    # Speed: 0.5 mps, Terrain: Flat, Mu: 75
    def get_file_id_and_loc(self):
        return "https://www.dropbox.com/scl/fi/j7y2ukxjv99c6ge357g39/robot_1_a1_0p5mps_flat_mu1_75_mu2_75_trial1_2024-09-09-14-37-30.bag?rlkey=osklprqilkxoxixes7xlflqgq&st=u3tyw691&dl=1", "Dropbox"

class QuadSDKDataset_A1_Charlie(QuadSDKDataset_A1):
    # Speed: 0.5 mps, Terrain: Flat, Mu: 100
    def get_file_id_and_loc(self):
        return "https://www.dropbox.com/scl/fi/3swmvf4zmqhqjtdi37e0i/robot_1_a1_0p5mps_flat_mu1_100_mu2_100_trial1_2024-09-09-15-04-36.bag?rlkey=pekugjjnexobbpb772qtoolky&st=tzxksa3k&dl=1", "Dropbox"
    
class QuadSDKDataset_A1_Delta(QuadSDKDataset_A1):
    # Speed: 0.5 mps, Terrain: Slope, Mu: 50
    def get_file_id_and_loc(self):
        return "https://www.dropbox.com/scl/fi/4syqkizbm9qthg7e35soh/robot_1_a1_0p5mps_slope_mu1_50_mu2_50_trial1_2024-09-09-15-29-14.bag?rlkey=rpztfeh8sufxmryb67rqgkzbx&st=0q0vsf76&dl=1", "Dropbox"

class QuadSDKDataset_A1_Echo(QuadSDKDataset_A1):
    # Speed: 0.5 mps, Terrain: Slope, Mu: 75
    def get_file_id_and_loc(self):
        return "https://www.dropbox.com/scl/fi/bkgasvttt5v7szcpjm350/robot_1_a1_0p5mps_slope_mu1_75_mu2_75_trial1_2024-09-09-15-46-45.bag?rlkey=eg33dr3brs03uvvsgmkkj0tn1&st=3n9ugtpc&dl=1", "Dropbox"

class QuadSDKDataset_A1_Foxtrot(QuadSDKDataset_A1):
    # Speed: 0.5 mps, Terrain: Slope, Mu: 100
    def get_file_id_and_loc(self):
        return "https://www.dropbox.com/scl/fi/i2rik324b5byqpedloaew/robot_1_a1_0p5mps_slope_mu1_100_mu2_100_trial1_2024-09-09-19-02-56.bag?rlkey=pm3z8wm2rxw7ubvtx88z1g827&st=022deyoe&dl=1", "Dropbox"
    
class QuadSDKDataset_A1_Golf(QuadSDKDataset_A1):
    # Speed: 0.5 mps, Terrain: Rough, Mu: 75
    def get_file_id_and_loc(self):
        return "https://www.dropbox.com/scl/fi/2k2v2bydphna2qx00e4oz/robot_1_a1_0p5mps_rough_mu1_75_mu2_75_trial1_2024-09-10-15-57-56.bag?rlkey=lc4e7826ro0qo4z9ri6dnokox&st=5j01yfle&dl=1", "Dropbox"
    
class QuadSDKDataset_A1_Hotel(QuadSDKDataset_A1):
    # Speed: 0.5 mps, Terrain: Rough, Mu: 100
    def get_file_id_and_loc(self):
        return "https://www.dropbox.com/scl/fi/luyqyl11x2nj3p52k0ddc/robot_1_a1_0p5mps_rough_mu1_100_mu2_100_trial1_2024-09-10-16-37-10.bag?rlkey=1j4hiy8f5qbfrf5ojwdyxvo81&st=woilhuqo&dl=1", "Dropbox"

class QuadSDKDataset_A1_India(QuadSDKDataset_A1):
    # Speed: 0.75 mps, Terrain: Flat, Mu: 50
    def get_file_id_and_loc(self):
        return "https://www.dropbox.com/scl/fi/f615giqzrgrt6rt8rxwn8/robot_1_a1_0p75mps_flat_mu1_50_mu2_50_trial1_2024-09-09-14-30-18.bag?rlkey=q4e35770xjjrm4c9p8dx85j5u&st=2ikap4wu&dl=1", "Dropbox"

class QuadSDKDataset_A1_Juliett(QuadSDKDataset_A1):
    # Speed: 0.75 mps, Terrain: Flat, Mu: 75
    def get_file_id_and_loc(self):
        return "https://www.dropbox.com/scl/fi/u2hlxjbb9dgg9ttq9kwl2/robot_1_a1_0p75mps_flat_mu1_75_mu2_75_trial1_2024-09-09-14-45-02.bag?rlkey=nyevs573wf273zefbvyvwvkq4&st=vlcau1i3&dl=1", "Dropbox"
    
class QuadSDKDataset_A1_Kilo(QuadSDKDataset_A1):
    # Speed: 0.75 mps, Terrain: Flat, Mu: 100
    def get_file_id_and_loc(self):
        return "https://www.dropbox.com/scl/fi/sp90ohnu0y2hrwv5hykol/robot_1_a1_0p75mps_flat_mu1_100_mu2_100_trial1_2024-09-09-15-11-33.bag?rlkey=y11sftmemzpymwfy2syijvt7y&st=kvu4ut66&dl=1", "Dropbox"

class QuadSDKDataset_A1_Lima(QuadSDKDataset_A1):
    # Speed: 0.75 mps, Terrain: Slope, Mu: 50
    def get_file_id_and_loc(self):
        return "https://www.dropbox.com/scl/fi/zbf9sxb5c6192xgtubb42/robot_1_a1_0p75mps_slope_mu1_50_mu2_50_trial1_2024-09-09-15-39-03.bag?rlkey=949w060ptq7dajgphly58ebou&st=o5eijrf5&dl=1", "Dropbox"    

class QuadSDKDataset_A1_Mike(QuadSDKDataset_A1):
    # Speed: 0.75 mps, Terrain: Slope, Mu: 75
    def get_file_id_and_loc(self):
        return "https://www.dropbox.com/scl/fi/7qwiwicd1x9h051rxyla0/robot_1_a1_0p75mps_slope_mu1_75_mu2_75_trial1_2024-09-09-16-11-33.bag?rlkey=7lmasurdpzlleojqltmikkin4&st=tj4hmxcx&dl=1", "Dropbox"
    
class QuadSDKDataset_A1_November(QuadSDKDataset_A1):
    # Speed: 0.75 mps, Terrain: Slope, Mu: 100
    def get_file_id_and_loc(self):
        return "https://www.dropbox.com/scl/fi/50rwpce47tushv8gb0oew/robot_1_a1_0p75mps_slope_mu1_100_mu2_100_trial1_2024-09-09-19-10-33.bag?rlkey=ajgb069ku2pp2uua49nv54c9q&st=rrd9u414&dl=1", "Dropbox"
    
class QuadSDKDataset_A1_Oscar(QuadSDKDataset_A1):
    # Speed: 0.75 mps, Terrain: Rough, Mu: 75
    def get_file_id_and_loc(self):
        return "https://www.dropbox.com/scl/fi/l2c5w2lccaaut1k77nm3i/robot_1_a1_0p75mps_rough_mu1_75_mu2_75_trial1_2024-09-10-16-11-05.bag?rlkey=m5j3ya1yuc9hjrx2axfjcw47d&st=hpf2lg6p&dl=1", "Dropbox"
    
class QuadSDKDataset_A1_Papa(QuadSDKDataset_A1):
    # Speed: 0.75 mps, Terrain: Rough, Mu: 100
    def get_file_id_and_loc(self):
        return "https://www.dropbox.com/scl/fi/wz9gn4zps1781eyrt8huz/robot_1_a1_0p75mps_rough_mu1_100_mu2_100_trial1_2024-09-10-16-49-42.bag?rlkey=vcebsg42ggxeemaagodgsb1yr&st=openvcsi&dl=1", "Dropbox"
    
class QuadSDKDataset_A1_Quebec(QuadSDKDataset_A1):
    # Speed: 1.00 mps, Terrain: Flat, Mu: 75
    def get_file_id_and_loc(self):
        return "https://www.dropbox.com/scl/fi/1lv5jxavljoep4dvlfwlb/robot_1_a1_1p0mps_flat_mu1_75_mu2_75_trial1_2024-09-09-14-57-16.bag?rlkey=9jryyynd345ad8lf4cmhdj247&st=7pm05bgq&dl=1", "Dropbox"

class QuadSDKDataset_A1_Romeo(QuadSDKDataset_A1):
    # Speed: 1.00 mps, Terrain: Flat, Mu: 100
    def get_file_id_and_loc(self):
        return "https://www.dropbox.com/scl/fi/sds8jnzftbk6f0vreyl0n/robot_1_a1_1p0mps_flat_mu1_100_mu2_100_trial1_2024-09-09-15-21-52.bag?rlkey=wxfyc8npy2jg0u7ws66ekece8&st=ipnftg46&dl=1", "Dropbox"

class QuadSDKDataset_A1_Sierra(QuadSDKDataset_A1):
    # Speed: 1.00 mps, Terrain: Slope, Mu: 75
    def get_file_id_and_loc(self):
        return "https://www.dropbox.com/scl/fi/hbx5gsdkp845c060uor6z/robot_1_a1_1p0mps_slope_mu1_75_mu2_75_trial1_2024-09-09-18-54-59.bag?rlkey=xtuooopimk4cq27nxqjz4wdlw&st=3jkrpqdr&dl=1", "Dropbox"
    
class QuadSDKDataset_A1_Tango(QuadSDKDataset_A1):
    # Speed: 1.00 mps, Terrain: Slope, Mu: 100
    def get_file_id_and_loc(self):
        return "https://www.dropbox.com/scl/fi/84lv882akicv37hmab0d7/robot_1_a1_1p0mps_slope_mu1_100_mu2_100_trial1_2024-09-09-19-24-12.bag?rlkey=761zudvru1ctdsde51ipx1onn&st=esumu7ot&dl=1", "Dropbox"
    
class QuadSDKDataset_A1_Uniform(QuadSDKDataset_A1):
    # Speed: 1.00 mps, Terrain: Rough, Mu: 50
    def get_file_id_and_loc(self):
        return "https://www.dropbox.com/scl/fi/fswpwmltqgwxa8pwb1rif/robot_1_a1_1p0mps_rough_mu1_50_mu2_50_trial1_2024-09-10-17-19-24.bag?rlkey=gqjn6458dt5rbmhv1klymm6xl&st=f9obhple&dl=1", "Dropbox"

# Go2
class QuadSDKDataset_Go2_Flat_Speed0_5_Mu_50(QuadSDKDataset_Go2):
    def get_file_id_and_loc(self):
        return "https://www.dropbox.com/scl/fi/qxsgvg9qhg6fmhkkrpdtp/robot_1_go2_0.5mps_mu50_mu250_trial1_2024-09-02-18-54-26.bag?rlkey=f9rjl7r4cvejupxharda64ctj&st=fy9g8fn2&dl=1", "Dropbox"