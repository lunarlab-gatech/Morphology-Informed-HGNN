import torch
from torch_geometric.data import Data, Dataset, HeteroData
from .graphParser import NormalRobotGraph, HeterogeneousRobotGraph
from rosbags.highlevel import AnyReader
from pathlib import Path
from torchvision.datasets.utils import download_file_from_google_drive
import os


class FlexibleDataset(Dataset):
    """
    Dataset class that can be used for a MLP, normal GNN, or 
    heterogeneous GNN.
    """

    # Used when initalizing the parent class instead of a proper
    # child classs
    notImplementedError = NotImplementedError(
        "Don't call this class directly, but use one of \
        the child classes in order to choose which dataset \
        sequence you want to load.")

    def __init__(self,
                 root: str,
                 urdf_path: Path,
                 ros_builtin_path: str,
                 urdf_to_desc_path: str,
                 data_format: str = 'gnn',
                 history_length: int = 1,
                 transform=None,
                 pre_transform=None,
                 pre_filter=None):
        """
        Parameters:
                history_length (int): The length of the history of inputs to use
                    for our graph entries.
                data_format (str): Either 'gnn', 'mlp', or 'heterogeneous_gnn'. 
                    Determines how the get() method returns data.
        """
        # Check for valid data format
        self.data_format = data_format
        if self.data_format != 'mlp' and self.data_format != 'gnn' and self.data_format != 'heterogeneous_gnn':
            raise ValueError(
                "Parameter 'data_format' must be 'gnn', 'mlp', or 'heterogeneous_gnn'."
            )

        # Setup the directories for raw and processed data
        self.root = root
        super().__init__(root, transform, pre_transform, pre_filter)

        # Get the first index and length of dataset
        info_path = Path(root, 'processed', 'info.txt')
        with open(info_path, 'r') as f:
            data = f.readline().split(" ")

            # The longer the history, technically the lower number of total dataset entries
            # we can use, and thus a lower dataset length.
            self.history_length = history_length
            self.length = int(data[0]) - (self.history_length - 1)
            if self.length <= 0:
                raise ValueError(
                    "Dataset has too few entries for the provided 'history_length'."
                )
            self.first_index = int(data[1])

        # Parse the robot graph from the URDF file
        if self.data_format == 'heterogeneous_gnn':
            self.robotGraph = HeterogeneousRobotGraph(urdf_path,
                                                      ros_builtin_path,
                                                      urdf_to_desc_path)
        else:
            self.robotGraph = NormalRobotGraph(urdf_path, ros_builtin_path,
                                               urdf_to_desc_path)

        # Make sure the URDF file we were given properly matches
        # what we are expecting
        passed_urdf_name = self.robotGraph.robot_urdf.name
        expected_name = self.get_expected_urdf_name()
        if passed_urdf_name != expected_name:
            raise ValueError(
                "Invalid URDF: \"" + passed_urdf_name +
                "\". This dataset was collected on the \"" + expected_name +
                "\" robot. Please pass in the URDF file for the \"" +
                expected_name + "\" robot, not for the \"" + passed_urdf_name +
                "\" robot.")

        # Get node name to index mapping
        self.urdf_name_to_graph_index = self.robotGraph.get_node_name_to_index_dict()

    def get_google_drive_file_id(self):
        """
        Method for child classes to choose which sequence to load;
        used if the dataset is
        """
        raise self.notImplementedError

    @property
    def raw_file_names(self):
        return ['data.bag']

    def download(self):
        download_file_from_google_drive(self.get_google_drive_file_id(),
                                        Path(self.root, 'raw'), "data.bag")

    @property
    def processed_file_names(self):
        """
        Make the list of all the processed 
        files we are expecting.
        """

        processed_file_names = []
        start_seq_id, end_seq_id = self.get_start_and_end_seq_ids()
        for i in range(start_seq_id, end_seq_id + 1):
            processed_file_names.append(str(i) + ".txt")
        processed_file_names.append("info.txt")
        return processed_file_names

    def process(self):
        raise self.notImplementedError

    def get_foot_node_indices_matching_labels(self):
        """
        This helper method returns the foot node indices that correspond 
        to the ground truth GRF labels contained in the "y" of each 
        Data object. The number x in the ith index tells us that 
        ground truth label i matches up with node x from the GNN output.

        For example, if the graph has 8 nodes, and there are only 4 
        ground truth labels, this method might return [0, 3, 4, 7].
        This means that the first ground truth label should match
        the output of the node at index 0, the second ground truth label
        should match the output of the node at index 3, and so on.

        Returns:
            foot_node_indices (list[int]) - List of the indices of
            the graph nodes that should output the ground truth labels 
            at the same corresponding index.
        """
        if self.foot_node_indices is None:
            raise self.notImplementedError
        else:
            return self.foot_node_indices

    def get_start_and_end_seq_ids(self):
        """
        Method for child classes to tell the processing method the
        numbers of the files they are looking for. It returns
        the seq id of the first ROS '/hardware_a1/joint_foot' 
        message, and the seq id of the last.
        """
        raise self.notImplementedError

    def get_expected_urdf_name(self):
        """
        Method for child classes to specify what URDF file they are
        looking for.
        """
        raise self.notImplementedError

    def len(self):
        return self.length

    def get_helper_mlp(self, idx):
        """
        Gets a Dataset entry if we are using an MLP model.
        """
        raise self.notImplementedError

    def get_helper_gnn(self, idx):
        """
        Get a dataset entry if we are using a GNN model.
        """
        raise self.notImplementedError

    def get_helper_heterogeneous_gnn(self, idx):
        """
        Get a dataset entry if we are using a Heterogeneous GNN model.
        """
        raise self.notImplementedError

    def get(self, idx):
        # Bounds check
        if idx < 0 or idx >= self.length:
            raise IndexError("Index value out of Dataset bounds.")

        # Return data in the proper format
        if self.data_format == 'gnn':
            return self.get_helper_gnn(idx)
        elif self.data_format == 'mlp':
            return self.get_helper_mlp(idx)
        elif self.data_format == 'heterogeneous_gnn':
            return self.get_helper_heterogeneous_gnn(idx)


class QuadSDKDataset(FlexibleDataset):
    """
    Dataset class for the simulated data provided through
    Quad-SDK and the Gazebo simulator.
    """

    def __init__(self,
                 root: str,
                 urdf_path: Path,
                 ros_builtin_path: str,
                 urdf_to_desc_path: str,
                 data_format: str = 'gnn',
                 history_length: int = 1,
                 transform=None,
                 pre_transform=None,
                 pre_filter=None):
        super().__init__(root, urdf_path, ros_builtin_path, urdf_to_desc_path,
                         data_format, history_length, transform, pre_transform, pre_filter)

        # Map urdf names to array indexes
        self.urdf_name_to_dataset_array_index = {
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
            'RL_foot_fixed': 1,
        }

        # Define the node names for the robot feet
        self.foot_urdf_names = [
            'FR_foot_fixed',
            'FL_foot_fixed',
            'RR_foot_fixed',
            'RL_foot_fixed',
        ]

        # Get the indices of the feet nodes in the robotGraph
        self.foot_node_indices = []
        for urdf_name in self.foot_urdf_names:
            self.foot_node_indices.append(
                self.urdf_name_to_graph_index[urdf_name])

        # Get the GT grf array indices that correspond to the feet nodes order
        self.gt_grf_array_indices = []
        for urdf_name in self.foot_urdf_names:
            self.gt_grf_array_indices.append(
                self.urdf_name_to_dataset_array_index[urdf_name])

        # Define the nodes that should recieve features
        self.joint_nodes_for_attributes = [
            'FL_hip_joint', 'FL_thigh_joint', 'FL_calf_joint',
            'FR_hip_joint', 'FR_thigh_joint', 'FR_calf_joint',
            'RL_hip_joint', 'RL_thigh_joint', 'RL_calf_joint',
            'RR_hip_joint', 'RR_thigh_joint', 'RR_calf_joint'
        ]
        self.base_nodes_for_attributes = ['floating_base']

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

        # Calculate accelerations using discrete methods
        av_2back, av_1back, av_curr = None, None, None
        jv_2back, jv_1back, jv_curr = None, None, None
        ts_2back, ts_1back, ts_curr = None, None, None
        for i in range(0, dataset_entries):
            with open(str(Path(self.processed_dir, str(i) + ".txt")), "r") as f:
                # Get the angular velocity, joint velocities, and timestamp info
                timestamp_info = f.readline().split(" ")[:-1]
                ts_curr = float(timestamp_info[2])

                line = f.readline()
                line = f.readline()
                av_curr = f.readline().split(" ")[:-1]

                line = f.readline()
                jv_curr = f.readline().split(" ")[:-1]

                # Calculate the angular acceleration and
                # joint accelerations (if possible) through
                # centered finite differences
                if i >= 2:
                    two_delta_ts = ts_curr - ts_2back
                    aa = []
                    ja = []
                    for j in range(0, len(av_curr)):
                        aa.append((float(av_curr[j]) - float(av_2back[j])) / two_delta_ts)
                    for j in range(0, len(jv_curr)):
                        ja.append((float(jv_curr[j]) - float(jv_2back[j])) / two_delta_ts)


                    with open(str(Path(self.processed_dir, str(i-1) + ".txt")), "a") as prev_f:
                        for val in aa:
                            prev_f.write(str(val) + " ")
                        prev_f.write('\n')
                        for val in ja:
                            prev_f.write(str(val) + " ")

                # Shift all of the measurements by 1
                av_2back = av_1back
                av_1back = av_curr
                av_curr = None
                jv_2back = jv_1back
                jv_1back = jv_curr
                jv_curr = None
                ts_2back = ts_1back
                ts_1back = ts_curr
                ts_curr = None

        # Write a txt file to save the dataset length & and first sequence index
        with open(os.path.join(self.processed_dir, "info.txt"), "w") as f:
            # First and last don't have accelerations, so ignore them
            f.write(str(dataset_entries - 2) + " " + str(1))

    def get_expected_urdf_name(self):
        return "a1"

    def get_start_and_end_seq_ids(self):
        return 0, 17530

    def load_data_at_dataset_seq(self, seq_num: int):
        """
        This helper function opens the file named "ros_seq"
        and loads the position, velocity, and effort information.

        Parameters:
            ros_seq (int): The sequence number of the ros message
                whose data should be loaded.
        """
        grfs, lin_acc, ang_vel, positions, velocities, torques, ang_acc, joint_accelerations = [], [], [], [], [], [], [], []
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
                ang_acc.append(float(line[i]))
            line = f.readline().split(" ")[:-1]
            for i in range(0, len(line)):
                joint_accelerations.append(float(line[i]))

        # Extract the ground truth Z GRF
        z_grfs = []
        for val in range(0, 4):
            start_index = val * 3
            z_grfs.append(grfs[start_index + 2])

        return lin_acc, ang_vel, positions, velocities, torques, z_grfs, ang_acc, joint_accelerations

    def load_data_sorted(self, seq_num: int):
        """
        Loads data from the dataset at the provided sequence number.
        However, the positions, velocities, joint acceleratinos, and torques 
        are sorted so that they match the order found in self.joint_nodes_for_attributes.
        Additionally, the Z_GRF_labels are sorted so it matches the order
        found in self.foot_urdf_names.
        """
        lin_acc, ang_vel, positions, velocities, torques, z_grfs, ang_acc, joint_acc = self.load_data_at_dataset_seq(seq_num)

        # Sort the joint information
        positions_sorted, velocities_sorted, torques_sorted, joint_acc_sorted  = [], [], [], []
        for urdf_node_name in self.joint_nodes_for_attributes:
            array_index = self.urdf_name_to_dataset_array_index[urdf_node_name]
            positions_sorted.append(positions[array_index])
            velocities_sorted.append(velocities[array_index])
            torques_sorted.append(torques[array_index])
            joint_acc_sorted.append(joint_acc[array_index])

        # Sort the Z_GRF_labels
        z_grfs_sorted = []
        for index in self.gt_grf_array_indices:
            z_grfs_sorted.append(z_grfs[index])

        return lin_acc, ang_vel, positions_sorted, velocities_sorted, torques_sorted, z_grfs_sorted, ang_acc, joint_acc_sorted

    def get_helper_mlp(self, idx):
        # Make the network inputs
        x = None

        # Load the rosbag information
        for i in range(0, self.history_length):
            lin_acc, ang_vel, positions, velocities, torques, ground_truth_labels, ang_acc, joint_acc = self.load_data_sorted(self.first_index + idx + i)
            tensor = torch.tensor((lin_acc + ang_vel + ang_acc + positions + velocities + joint_acc + torques), dtype=torch.float64).unsqueeze(0)
            if x is None:
                x = tensor
            else:
                x = torch.cat((x, tensor), 0)
        if len(x.size()) > 1:
            x = torch.flatten(torch.transpose(x, 0, 1), 0, 1)

        # Create the ground truth lables
        lin_acc, ang_vel, positions, velocities, torques, ground_truth_labels, ang_acc, joint_acc = self.load_data_sorted(self.first_index + idx + self.history_length - 1)
        y = torch.tensor(ground_truth_labels, dtype=torch.float64)
        return x, y

    def get_helper_gnn(self, idx):
        # Create a note feature matrix
        x = torch.ones((self.robotGraph.get_num_nodes(), 4 * self.history_length), dtype=torch.float64)

        # For each dataset entry we include in the history
        for j in range(0, self.history_length):
            # Load the data for this entry
            lin_acc, ang_vel, positions, velocities, torques, z_grfs, ang_acc, joint_acc = self.load_data_sorted(self.first_index + idx + j)

            # For each joint specified
            for i, urdf_node_name in enumerate(self.joint_nodes_for_attributes):

                # Find the index of this particular node
                node_index = self.urdf_name_to_graph_index[urdf_node_name]

                # Add the features to x matrix
                x[node_index, 0*self.history_length+j] = positions[i]
                x[node_index, 1*self.history_length+j] = velocities[i]
                x[node_index, 2*self.history_length+j] = joint_acc[i]
                x[node_index, 3*self.history_length+j] = torques[i]

        # Create the edge matrix
        self.edge_matrix = self.robotGraph.get_edge_index_matrix()
        self.edge_matrix_tensor = torch.tensor(self.edge_matrix,
                                               dtype=torch.long)

        # Create the labels
        lin_acc, ang_vel, positions, velocities, torques, z_grfs, ang_acc, joint_acc = self.load_data_sorted(
            self.first_index + idx + self.history_length - 1)
        y = torch.tensor(z_grfs, dtype=torch.float64)

        # Create the graph
        graph = Data(x=x, edge_index=self.edge_matrix_tensor,
                     y=y, num_nodes=self.robotGraph.get_num_nodes())
        return graph

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
        raise "17tvm0bmipTpueehUNQ-hJ8w5arc79q0M"

class QuadSDKDataset_A1Speed1_0(QuadSDKDataset):
    def get_google_drive_file_id(self):
        raise "1qSdm8Rm6UazwhzCV5DfMHF0AoyKNrthf"

class QuadSDKDataset_A1Speed1_5FlippedOver(QuadSDKDataset):
    def get_google_drive_file_id(self):
        raise "1h5CN-IIJlLnMvWp0sk5Ho-hiJq2NMqCT"