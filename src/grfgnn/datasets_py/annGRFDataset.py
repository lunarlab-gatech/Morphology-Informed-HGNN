import torch
import os
from .flexibleDataset import FlexibleDataset
import csv
from pathlib import Path
import pandas as pd

class annGRFDataset(FlexibleDataset):
    """
    Dataset class for the datasets used in the paper 
    "Artificial neural network-based ground reaction 
    force estimation and learning for dynamic-legged 
    robot systems". Specifically designed for the First
    Stage MLP dataset.
    """

    def __init__(self,
                 root: str,
                 urdf_path: Path,
                 ros_builtin_path: str,
                 urdf_to_desc_path: str,
                 data_format: str = 'mlp',
                 history_length: int = 3,
                 transform=None,
                 pre_transform=None,
                 pre_filter=None):
        super().__init__(root, urdf_path, ros_builtin_path, urdf_to_desc_path,
                         data_format, history_length, transform, pre_transform, pre_filter)
        
        self.foot_node_indices = [0]

        # Make sure users know that only a history length of 3 can be used
        if history_length != 3:
            raise ValueError("The ANN dataset currently only supports a history_length of 3.")

        # Parent uses history length parameter to adjust size of dataset
        # ANN entries already include history size of three, so no need
        # to do this. Reset the length here
        info_path = Path(root, 'processed', 'info.txt')
        with open(info_path, 'r') as f:
            data = f.readline().split(" ")
            self.history_length = history_length
            self.length = int(data[0])

    def process(self):
        dataset_entries = 0
        for i in range(1, 3):
            path_to_csv = str(Path(self.root, "raw", "Datasets for GRF estimation (First stage MLP) - " + str(i) + ".csv"))
            with open(path_to_csv, newline='') as csvfile:
                spamreader = csv.reader(csvfile, delimiter=' ', quotechar='|')
                for row in spamreader:
                    # Save the important data
                    with open(str(Path(self.processed_dir,
                                    str(dataset_entries) + ".txt")), "w") as f:
                            for val in row:
                                f.write(str(val) + " ")

                    # Track how many entries we have
                    dataset_entries += 1

        # Write a txt file to save the dataset length & and first sequence index
        with open(os.path.join(self.processed_dir, "info.txt"), "w") as f:
            # First and last don't have accelerations, so ignore them
            f.write(str(dataset_entries) + " " + str(0))

    def get_expected_urdf_name(self):
        return "a1"

    def load_data_at_dataset_seq(self, seq_num: int):
        """
        This helper function opens the file named "ros_seq"
        and loads the position, velocity, and effort information.

        Parameters:
            ros_seq (int): The sequence number of the ros message
                whose data should be loaded.
        """
        data = []
        with open(os.path.join(self.processed_dir,
                               str(seq_num) + ".txt"), 'r') as f:
            line = f.readline().split(",")
            for i in range(0, len(line)):
                data.append(float(line[i]))
        return data

    def get_helper_mlp(self, idx):
        data = self.load_data_at_dataset_seq(self.first_index + idx)

        # Make the network inputs
        x = torch.tensor(data[0:14], dtype=torch.float64)

        # Create the ground truth lables
        y = torch.tensor(data[14], dtype=torch.float64)
        return x, y
        
    @property
    def raw_file_names(self):
        return []

    def download(self):
        pass

    def get_start_and_end_seq_ids(self):
        return 0, 100