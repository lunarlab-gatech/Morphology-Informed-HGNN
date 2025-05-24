from .flexibleDataset import FlexibleDataset
import numpy as np
import os
from pathlib import Path
from rosbags.highlevel import AnyReader
from rosbags.typesys import Stores, get_types_from_msg, get_typestore
import scipy.io as sio

class CustomDataset(FlexibleDataset):
    """
    Dataset class for loading the dataset collected of the
    UniTree Go2 robot at Georgia Tech's Human Augmentation
    Core Facility by the Lunar Lab.
    """

    # ========================= DOWNLOADING ==========================
    def get_downloaded_dataset_file_names(self):
        return ["rosbag.db3.zstd", "metadata.yaml"]

    # ========================= PROCESSING ===========================
    def process(self):
        # Load the path to the downoaded file
        path_to_file = Path(self.root, 'raw', 'data.<YOUR_EXTENSION_HERE>')

    
        # Get all files in 'rosmsgs' directory that are of type .msg
        filepaths = []
        for dirpath, _, filenames in os.walk(Path(__file__).parent / Path('rosmsgs')):
            for filename in filenames:
                file_split = filename.split('.')
                if len(file_split) >= 2 and file_split[1] == "msg":
                    filepaths.append(os.path.join(dirpath, filename))

        # Register custom messages into typestore
        typestore = get_typestore(Stores.ROS2_FOXY)
        add_types = {}
        for filepath in filepaths:
            msgpath = Path(filepath)
            name = msgpath.relative_to(msgpath.parents[1]).with_suffix('')
            if 'msg' not in name.parts:
                name = name.parent / 'msg' / name.name
            msgdef = msgpath.read_text(encoding='utf-8')
            add_types.update(get_types_from_msg(msgdef, name))
        typestore.register(add_types)

        # Create arrays for all of the data types
        timestamps = np.empty((0, 3), dtype=np.float64)
        imu_acc = np.empty((0, 3), dtype=np.float64)
        imu_omega = np.empty((0, 3), dtype=np.float64)
        q = np.empty((0, 12), dtype=np.float64)
        qd = np.empty((0, 12), dtype=np.float64) 
        tau = np.empty((0, 12), dtype=np.float64) 
        f_p = np.empty((0, 12), dtype=np.float64)
        f_v = np.empty((0, 12), dtype=np.float64)
        F = np.empty((0, 4), dtype=np.float64)
        
        # NOTE: Robot Pose information (including vel/acc) is either
        # in the VICON data and/or could be extracted from VICON. Also,
        # joint acceleration is contained within this rosbag. Feel free
        # to extract this information for use with dynamics model or
        # otherwise.

        # Open Rosbag and extract pressure sensor data
        pressure_data = []
        with AnyReader([self.rosbagPath], default_typestore=typestore) as reader:
            connections = [x for x in reader.connections if x.topic == '/lowstate']
            for connection, timestamp, rawdata in reader.messages(connections=connections):
                msg = reader.deserialize(rawdata, connection.msgtype)

                # Extract timestamps (Assume time message was recorded is same as time sent for now)
                timestamps = np.concatenate((timestamps, np.array([[timestamp, timestamp, timestamp]], dtype=np.float64)), axis=0)
                
                # Extract IMU data
                lin_acc = msg.imu_state.accelerometer
                ang_vel = msg.imu_state.gyroscope
                imu_acc = np.concatenate((imu_acc, np.array([[lin_acc[0], lin_acc[1], lin_acc[2]]], dtype=np.float64)), axis=0)
                imu_omega = np.concatenate((imu_omega, np.array([[ang_vel[0], ang_vel[1], ang_vel[2]]], dtype=np.float64)), axis=0)

                # Extract joint information
                joint_data = msg.motor_state
                positions, velocities, efforts = [], [], []
                for i in range(0, 12):
                    positions.append(joint_data.q)
                    velocities.append(joint_data.dq)
                    efforts.append(joint_data.tau_est)
                q = np.concatenate((q, np.array(positions, dtype=np.float64)), axis=0)
                qd = np.concatenate((qd, np.array(velocities, dtype=np.float64)), axis=0)
                tau = np.concatenate((tau, np.array(efforts, dtype=np.float64)), axis=0)

                # Extract the ground truth GRF data
                grf_vec = grf_data.vectors
                grf_array = np.array([[ grf_vec[0].x, grf_vec[0].y, grf_vec[0].z,
                                    grf_vec[1].x, grf_vec[1].y, grf_vec[1].z,
                                    grf_vec[2].x, grf_vec[2].y, grf_vec[2].z,
                                    grf_vec[3].x, grf_vec[3].y, grf_vec[3].z]], dtype=np.float64)
                F = np.concatenate((F, grf_array), axis=0)

                # Track how many entries we have
                dataset_entries += 1

                pressure_data.append(msg.foot_force)

        # Convert into a dataframe
        pressure_data = np.array(pressure_data)
        titles = ["Pressure Sensor (Index #1)", "Pressure Sensor (Index #2)", 
                  "Pressure Sensor (Index #3)", "Pressure Sensor (Index #4)"]
        data_dict = {"Timestamp (sec)": np.array(timestamps),
                     titles[0]: pressure_data[:,0],
                     titles[1]: pressure_data[:,1],
                     titles[2]: pressure_data[:,2],
                     titles[3]: pressure_data[:,3]}
        df = pd.DataFrame(data_dict)

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

class HACLabDataset_SamePlaceMovingSlightly(CustomDataset):
    def get_file_ids_and_loc(self):
        return ["https://www.dropbox.com/scl/fi/oskpkdf5zo3n1z9i2px6i/rosbag2_2025_01_24-18_34_00_0.db3.zstd?rlkey=2jknmbmr121c35gpleu8sy6zp&st=cuco7lpk&dl=1", "https://www.dropbox.com/scl/fi/ezo2isyllrvfeyuxobstl/metadata.yaml?rlkey=9vcba5u2u46calamujl5374mx&st=dehxmxx5&dl=1"], "Dropbox"
    
