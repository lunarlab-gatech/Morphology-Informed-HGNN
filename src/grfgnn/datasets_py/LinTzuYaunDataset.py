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

        # Get the number of dataset entries in the file
        dataset_entries = mat_data['contacts'].shape[0]

        # Save each entry into its own text file
        for i in range(0, dataset_entries):

            # Save the important data
            with open(str(Path(self.processed_dir, str(i) + ".txt")), "w") as f:
                arrays = []

                # Save the contact labels
                arrays.append(mat_data['contacts'][i])

                # Add on the 'F' data (not sure what this is, potentially
                # could be force measurements, but not sure if this is from
                # foot sensors, or calculated after the fact, or if it's
                # something else entirely)
                arrays.append(mat_data['F'][i])

                # Add on the IMU data
                arrays.append(mat_data['imu_acc'][i])
                arrays.append(mat_data['imu_omega'][i])

                # Add on the joint data
                arrays.append(mat_data['q'][i])
                arrays.append(mat_data['qd'][i])
                arrays.append(mat_data['tau_est'][i])

                # Add on the foot pos/vel data
                arrays.append(mat_data['p'][i])
                arrays.append(mat_data['v'][i])

                for array in arrays:
                    for val in array:
                        f.write(str(val) + " ")
                    f.write('\n')

        # Write a txt file to save the dataset length & and first sequence index
        with open(str(Path(self.processed_dir, "info.txt")), "w") as f:
            f.write(str(dataset_entries) + " " + str(0) + " " + self.get_google_drive_file_id())

    # ============= DATA SORTING ORDER AND MAPPINGS ==================
    def get_base_node_sorted_order(self) -> list[str]:
        return ['floating_base']

    def get_joint_node_sorted_order(self) -> list[str]:
        return ['FR_hip_joint', 'FR_thigh_joint', 'FR_calf_joint',
                'FL_hip_joint', 'FL_thigh_joint',  'FL_calf_joint',
                'RR_hip_joint', 'RR_thigh_joint', 'RR_calf_joint',
                'RL_hip_joint', 'RL_thigh_joint', 'RL_calf_joint']

    def get_foot_node_sorted_order(self) -> list[str]:
        return ['FR_foot_fixed',
               'FL_foot_fixed',
               'RR_foot_fixed',
               'RL_foot_fixed']
    
    def get_urdf_name_to_dataset_array_index(self) -> dict:

        # Order of joint data can be found here: https://github.com/mit-biomimetics/Cheetah-Software/blob/master/documentation/getting_started.md
        # Note: Because they refer to the first joint as the abduction/adduction joint, and the second joint as the hip joint, this means that
        # our "hip_joint" corresponds to their abduction/adduction joint, and our "thigh_joint" corresponds to what they call the hip joint.
        # Dataset order of label data can be found here: https://github.com/UMich-CURLY/deep-contact-estimator/blob/master/utils/mat2numpy.py
        return {
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
            'RL_foot_fixed': 3}

    # ===================== DATASET PROPERTIES =======================
    def get_expected_urdf_name(self):
        return "miniCheetah"

    # ======================== DATA LOADING ==========================
    def load_data_at_dataset_seq(self, seq_num: int):
        contact_labels, lin_acc, ang_vel, j_p, j_v, j_T, f_p, f_v = [], [], [], [], [], [], [], []
        with open(str(Path(self.processed_dir, str(seq_num) + ".txt")), 'r') as f:

            line = f.readline().split(" ")[:-1]
            for i in range(0, len(line)):
                contact_labels.append(float(line[i]))
    
            # Skip the F data
            line = f.readline().split(" ")[:-1]

            line = f.readline().split(" ")[:-1]
            for i in range(0, len(line)):
                lin_acc.append(float(line[i]))
            line = f.readline().split(" ")[:-1]
            for i in range(0, len(line)):
                ang_vel.append(float(line[i]))
            line = f.readline().split(" ")[:-1]
            for i in range(0, len(line)):
                j_p.append(float(line[i]))
            line = f.readline().split(" ")[:-1]
            for i in range(0, len(line)):
                j_v.append(float(line[i]))
            line = f.readline().split(" ")[:-1]
            for i in range(0, len(line)):
                j_T.append(float(line[i]))
            line = f.readline().split(" ")[:-1]
            for i in range(0, len(line)):
                f_p.append(float(line[i]))
            line = f.readline().split(" ")[:-1]
            for i in range(0, len(line)):
                f_v.append(float(line[i]))

        # Convert them all to numpy arrays
        lin_acc = np.array(lin_acc)
        ang_vel = np.array(ang_vel)
        j_p = np.array(j_p)
        j_v = np.array(j_v)
        j_T = np.array(j_T)
        f_p = np.array(f_p)
        f_v = np.array(f_v)
        contact_labels = np.array(contact_labels)

        return lin_acc, ang_vel, j_p, j_v, None, f_p, f_v, contact_labels, None, None
    
# ================================================================
# ===================== DATASET SEQUENCES ========================
# ================================================================

class LinTzuYaunDataset_air_jumping_gait(LinTzuYaunDataset):
    def get_google_drive_file_id(self):
        return "17h4kMUKMymG_GzTZTMHPgj-ImDKZMg3R"
    
class LinTzuYaunDataset_air_walking_gait(LinTzuYaunDataset):
    def get_google_drive_file_id(self):
        return "17c_E-S_yTeeV_DCmcgVT7_J90cRIwg0z"
    
class LinTzuYaunDataset_asphalt_road(LinTzuYaunDataset):
    def get_google_drive_file_id(self):
        return "1jty0yqd7gywNJEkS_V2hivZ-79BGuCgA"
    
class LinTzuYaunDataset_old_asphalt_road(LinTzuYaunDataset):
    def get_google_drive_file_id(self):
        return "1Y4SHVLqQKQ14leBdpfEQv1Tq5uQUIEK8"
    
class LinTzuYaunDataset_concrete_right_circle(LinTzuYaunDataset):
    def get_google_drive_file_id(self):
        return "1NnEnd0PFFT6XozErUNi3ORGVSuFkyjeJ"
    
class LinTzuYaunDataset_concrete_pronking(LinTzuYaunDataset):
    def get_google_drive_file_id(self):
        return "1XWdEIKUtFKmZd9W5M7636-HVdusqglhd"

class LinTzuYaunDataset_concrete_left_circle(LinTzuYaunDataset):
    def get_google_drive_file_id(self):
        return "1K9hUMqc0oBCv6VtgS0rYXbRjq9XiFOv5"
    
class LinTzuYaunDataset_concrete_galloping(LinTzuYaunDataset):
    def get_google_drive_file_id(self):
        return "1oInoPLowARNsL0h_qPVgjLCLICR7zw7W"
    
class LinTzuYaunDataset_concrete_difficult_slippery(LinTzuYaunDataset):
    def get_google_drive_file_id(self):
        return "1i7MNbJNCBkIfW5TOU94YHnb5G0jXkSAf"
    
class LinTzuYaunDataset_forest(LinTzuYaunDataset):
    def get_google_drive_file_id(self):
        return "1qMriGIWAUXFN3a-ewfdVAZlDsi_jZRNi"
    
class LinTzuYaunDataset_grass(LinTzuYaunDataset):
    def get_google_drive_file_id(self):
        return "1yVRmhPZN6wpKhsT947Jkr8mlels8WM7m"
    
class LinTzuYaunDataset_middle_pebble(LinTzuYaunDataset):
    def get_google_drive_file_id(self):
        return "128UAFroCGekx-Ibk-zEAGYlq8mekdzOI"
    
class LinTzuYaunDataset_rock_road(LinTzuYaunDataset):
    def get_google_drive_file_id(self):
        return "1Hyo9UQkmAGrA0r49jZgVTAOe40SgnlfU"
    
class LinTzuYaunDataset_sidewalk(LinTzuYaunDataset):
    def get_google_drive_file_id(self):
        return "1D1vAmruuZE5KQH8gA_pDhfETHPMhiu2c"
    
class LinTzuYaunDataset_small_pebble(LinTzuYaunDataset):
    def get_google_drive_file_id(self):
        return "1cmjzHD9CKAXmKxZkDbPsEPKGvDI5Grec"