from .flexibleDataset import FlexibleDataset
import scipy.io as sio
from pathlib import Path
import numpy as np

class LinTzuYaunDataset(FlexibleDataset):
    """
    Dataset class for the MIT Mini Cheetah Contact Dataset
    found at https://github.com/UMich-CURLY/deep-contact-estimator .
    """

    # ========================= DOWNLOADING ==========================
    def get_downloaded_dataset_file_name(self):
        return "data.mat"

    # ========================= PROCESSING ===========================
    def process(self):
        # Load the .mat file
        path_to_mat = Path(self.root, 'raw', 'data.mat')
        mat_data = sio.loadmat(path_to_mat)

        # Save a copy in the processed directory
        sio.savemat(Path(self.root, 'processed', 'data.mat'), mat_data)

        # Get the number of dataset entries in the file
        dataset_entries = mat_data['contacts'].shape[0]

        # Write a txt file to save the dataset length & and first sequence index
        with open(str(Path(self.processed_dir, "info.txt")), "w") as f:
            file_id, loc = self.get_file_id_and_loc()
            f.write(str(dataset_entries) + " " + file_id)

    # ============= DATA SORTING ORDER AND MAPPINGS ==================
    def get_urdf_name_to_dataset_array_index(self) -> dict:

        # Order of joint data can be found here: https://github.com/mit-biomimetics/Cheetah-Software/blob/master/documentation/getting_started.md
        # Note: Because they refer to the first joint as the abduction/adduction joint, and the second joint as the hip joint, this means that
        # our "hip_joint" corresponds to their abduction/adduction joint, and our "thigh_joint" corresponds to what they call the hip joint.
        # Dataset order of label data can be found here: https://github.com/UMich-CURLY/deep-contact-estimator/blob/master/utils/mat2numpy.py
        return {
            'floating_base': 0,

            'RL_hip_joint': 9,
            'RL_thigh_joint': 10,
            'RL_calf_joint': 11,
            'FL_hip_joint': 3,
            'FL_thigh_joint': 4,
            'FL_calf_joint': 5,
            'RR_hip_joint': 6,
            'RR_thigh_joint': 7,
            'RR_calf_joint': 8,
            'FR_hip_joint': 0,
            'FR_thigh_joint': 1,
            'FR_calf_joint': 2,

            'RL_foot_fixed': 3,
            'FL_foot_fixed': 1,
            'RR_foot_fixed': 2,
            'FR_foot_fixed': 0}

    # ===================== DATASET PROPERTIES =======================
    def get_expected_urdf_name(self):
        return "miniCheetah"

    # ======================== DATA LOADING ==========================
    def load_data_at_dataset_seq(self, seq_num: int):
        """
        The units of the return values are as follows:
        - lin_acc (meters/sec^2)
        - ang_vel (rad/sec)
        - j_p (rad)
        - j_v (rad/sec)
        - f_p (meters, represented in robot's hip frame) 
        - f_v (meters/sec, represented in robot's hip frame)
        - contact_labels (no units)
        """

        # Convert them all to numpy arrays
        lin_acc = np.array(self.mat_data['imu_acc'][seq_num:seq_num+self.history_length]).reshape(self.history_length, 3)
        ang_vel = np.array(self.mat_data['imu_omega'][seq_num:seq_num+self.history_length]).reshape(self.history_length, 3)
        j_p = np.array(self.mat_data['q'][seq_num:seq_num+self.history_length]).reshape(self.history_length, 12)
        j_v = np.array(self.mat_data['qd'][seq_num:seq_num+self.history_length]).reshape(self.history_length, 12)
        j_T = np.array(self.mat_data['tau_est'][seq_num:seq_num+self.history_length]).reshape(self.history_length, 12)
        f_p = np.array(self.mat_data['p'][seq_num:seq_num+self.history_length]).reshape(self.history_length, 12)
        f_v = np.array(self.mat_data['v'][seq_num:seq_num+self.history_length]).reshape(self.history_length, 12)
        contact_labels = np.squeeze(np.array(self.mat_data['contacts'][seq_num:seq_num+self.history_length])[-1])

        return lin_acc, ang_vel, j_p, j_v, None, f_p, f_v, contact_labels, None, None, None
    
# ================================================================
# ===================== DATASET SEQUENCES ========================
# ================================================================

class LinTzuYaunDataset_air_jumping_gait(LinTzuYaunDataset):
    def get_file_id_and_loc(self):
        return "17h4kMUKMymG_GzTZTMHPgj-ImDKZMg3R", "Google"
    
class LinTzuYaunDataset_air_walking_gait(LinTzuYaunDataset):
    def get_file_id_and_loc(self):
        return "17c_E-S_yTeeV_DCmcgVT7_J90cRIwg0z", "Google"
    
class LinTzuYaunDataset_asphalt_road(LinTzuYaunDataset):
    def get_file_id_and_loc(self):
        return "1jty0yqd7gywNJEkS_V2hivZ-79BGuCgA", "Google"
    
class LinTzuYaunDataset_old_asphalt_road(LinTzuYaunDataset):
    def get_file_id_and_loc(self):
        return "1Y4SHVLqQKQ14leBdpfEQv1Tq5uQUIEK8", "Google"
    
class LinTzuYaunDataset_concrete_right_circle(LinTzuYaunDataset):
    def get_file_id_and_loc(self):
        return "1NnEnd0PFFT6XozErUNi3ORGVSuFkyjeJ", "Google"
    
class LinTzuYaunDataset_concrete_pronking(LinTzuYaunDataset):
    def get_file_id_and_loc(self):
        return "1XWdEIKUtFKmZd9W5M7636-HVdusqglhd", "Google"

class LinTzuYaunDataset_concrete_left_circle(LinTzuYaunDataset):
    def get_file_id_and_loc(self):
        return "1K9hUMqc0oBCv6VtgS0rYXbRjq9XiFOv5", "Google"
    
class LinTzuYaunDataset_concrete_galloping(LinTzuYaunDataset):
    def get_file_id_and_loc(self):
        return "1oInoPLowARNsL0h_qPVgjLCLICR7zw7W", "Google"
    
class LinTzuYaunDataset_concrete_difficult_slippery(LinTzuYaunDataset):
    def get_file_id_and_loc(self):
        return "1i7MNbJNCBkIfW5TOU94YHnb5G0jXkSAf", "Google"
    
class LinTzuYaunDataset_forest(LinTzuYaunDataset):
    def get_file_id_and_loc(self):
        return "1qMriGIWAUXFN3a-ewfdVAZlDsi_jZRNi", "Google"
    
class LinTzuYaunDataset_grass(LinTzuYaunDataset):
    def get_file_id_and_loc(self):
        return "1yVRmhPZN6wpKhsT947Jkr8mlels8WM7m", "Google"
    
class LinTzuYaunDataset_middle_pebble(LinTzuYaunDataset):
    def get_file_id_and_loc(self):
        return "128UAFroCGekx-Ibk-zEAGYlq8mekdzOI", "Google"
    
class LinTzuYaunDataset_rock_road(LinTzuYaunDataset):
    def get_file_id_and_loc(self):
        return "1Hyo9UQkmAGrA0r49jZgVTAOe40SgnlfU", "Google"
    
class LinTzuYaunDataset_sidewalk(LinTzuYaunDataset):
    def get_file_id_and_loc(self):
        return "1D1vAmruuZE5KQH8gA_pDhfETHPMhiu2c", "Google"
    
class LinTzuYaunDataset_small_pebble(LinTzuYaunDataset):
    def get_file_id_and_loc(self):
        return "1cmjzHD9CKAXmKxZkDbPsEPKGvDI5Grec", "Google"