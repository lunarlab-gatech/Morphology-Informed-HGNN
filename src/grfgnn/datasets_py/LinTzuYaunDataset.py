from .flexibleDataset import FlexibleDataset
import scipy.io as sio
from pathlib import Path

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
        print(dataset_entries)

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
            f.write(str(dataset_entries) + " " + str(0))

    # ============= DATA SORTING ORDER AND MAPPINGS ==================
    def get_base_node_sorted_order(self) -> list[str]:
        raise self.notImplementedError

    def get_joint_node_sorted_order(self) -> list[str]:
        raise self.notImplementedError

    def get_foot_node_sorted_order(self) -> list[str]:
        raise self.notImplementedError
    
    def get_urdf_name_to_dataset_array_index(self) -> dict:
        raise self.notImplementedError

    # ===================== DATASET PROPERTIES =======================
    def get_expected_urdf_name(self):
        return "miniCheetah"

    # ======================== DATA LOADING ==========================
    def load_data_at_dataset_seq(self, seq_num: int):
        contact_labels, lin_acc, ang_vel, j_p, j_v, j_T, f_p, f_v = [], [], [], [], [], []
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

        return lin_acc, ang_vel, j_p, j_v, j_T, f_p, f_v, contact_labels
    
# ================================================================
# ===================== DATASET SEQUENCES ========================
# ================================================================

class LinTzuYaunDataset_air_jumping_gait(LinTzuYaunDataset):
    def get_google_drive_file_id(self):
        return "17h4kMUKMymG_GzTZTMHPgj-ImDKZMg3R"