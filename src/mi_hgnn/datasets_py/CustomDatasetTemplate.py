from .flexibleDataset import FlexibleDataset
import scipy.io as sio
from pathlib import Path
import numpy as np

class CustomDataset(FlexibleDataset):


    # ========================= DOWNLOADING ==========================
    def get_downloaded_dataset_file_names(self):
        """
        Type the names of the file extension of your dataset sequence files here!
        """
        return ["data.<YOUR_EXTENSION_HERE>"]

    # ========================= PROCESSING ===========================
    def process(self):
        # Load the path to the downoaded file
        path_to_file = Path(self.root, 'raw', 'data.<YOUR_EXTENSION_HERE>')

        # TODO: Convert into a MATLAB data dictionary format here!
        mat_data = None

        # Make sure to save it at this location
        sio.savemat(Path(self.root, 'processed', 'data.mat'), mat_data)

        # TODO: Get the number of dataset entries in the file
        dataset_entries = None

        # Write a txt file to save the dataset length & and first sequence index
        with open(str(Path(self.processed_dir, "info.txt")), "w") as f:
            file_ids, loc = self.get_file_ids_and_loc()
            f.write(str(dataset_entries) + " " + file_ids[0])

    # ============= DATA SORTING ORDER AND MAPPINGS ==================
    def get_urdf_name_to_dataset_array_index(self) -> dict:
        """
        Implement this function to tell `FlexibleDataset` how 
        the data returned by `load_data_at_dataset_seq()` corresponds
        to the joints in the robot URDF file.

        Traditionally a robot only has one base node, so it should get a value
        of 0. Next, type the name of each leg joint in the URDF file, and add
        the index of its value in the corresponding joint arrays returned by
        load_data_at_dataset_seq(). Do the same for the joints in the URDF
        representing a fixed foot, with the indices of their values in the foot 
        position and foot velocity arrays.
        """

        return {
            '<URDF_BASE_NODE>': 0,

            '<URDF_JOINT_NODE>': 2,
            '<URDF_JOINT_NODE2>': 0,
            '<URDF_JOINT_NODE3>': 1,

            '<URDF_FOOT_NODE>': 1,
            '<URDF_FOOT_NODE2>': 0,
            }

    # ===================== DATASET PROPERTIES =======================
    def get_expected_urdf_name(self):
        return "<EXPECTED_URDF_NAME_HERE>"

    # ======================== DATA LOADING ==========================
    def load_data_at_dataset_seq(self, seq_num: int):
        """
        When this function is called, the .mat file data saved in process()
        is available at self.mat_data.

        For information on the expected format of these variables, see the
        load_data_at_dataset_seq() function defition in flexibleDataset.py.
        """

        # TODO: Load the data as numpy arrays, and don't forget to incorporate self.history_length
        # to load a history of measurments.
        lin_acc = None
        ang_vel = None
        j_p = None
        j_v = None
        j_T = None
        f_p = None
        f_v = None
        contact_labels = None
        r_p = None
        r_o = None
        timestamps = None
        # Note, if you don't have data for a specific return value, just return None, 
        # and `FlexibleDataset` will know not to use it if it is not required.

        return lin_acc, ang_vel, j_p, j_v, j_T, f_p, f_v, contact_labels, r_p, r_o, timestamps
    
# ================================================================
# ===================== DATASET SEQUENCES ========================
# ================================================================

class CustomDataset_sequence1(CustomDataset):
    """
    To load a dataset sequence from Google, first upload the corresponding file on Google Drive, set "General Access"
    to "Anyone with the link", and then copy the link. Paste the link, and then extract the string between the text of
    '/file/d/' and '/view?usp=sharing'. Take this string, and paste it as the first return argument below.
    """
    def get_file_ids_and_loc(self):
        return ["<Your_String_Here>"], "Google"
    
class CustomDataset_sequence2(CustomDataset):
    """
    To load a dataset sequence from Dropbox, first you'll need to upload the corresponding file on Dropbox and  
    generate a link for viewing. Make sure that access is given to anyone with the link, and that this permission won't
    expire, doesn't require a password, and allows for downloading. Finally, copy and paste the link as the first return
    argument below, but change the last number from 0 to 1 (this tells Dropbox to send the raw file, instead of a webpage).
    """
    def get_file_ids_and_loc(self):
        return ["<Your_Link_Here>"], "Dropbox"
    
"""
Create classes for each of your sequences...
"""