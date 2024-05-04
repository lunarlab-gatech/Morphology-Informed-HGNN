from grfgnn.datasets import visualize_graph, CerberusTrackDataset, CerberusStreetDataset, CerberusCampusDataset, Go1SimulatedDataset
from grfgnn.graphParser import NormalRobotGraph
from pathlib import Path
from grfgnn.gnnLightning import evaluate_model_and_visualize
import os
from rosbags.highlevel import AnyReader
import numpy as np


def dataset_create():
    path_to_go1_urdf = Path(Path('.').parent, 'urdf_files', 'Go1',
                            'go1.urdf').absolute()
    path_to_cerberus_street = Path(
        Path('.').parent, 'datasets', 'cerberus_street').absolute()
    path_to_cerberus_track = Path(
        Path('.').parent, 'datasets', 'cerberus_track').absolute()
    path_to_cerberus_campus = Path(
        Path('.').parent, 'datasets', 'cerberus_campus').absolute()
    path_to_xiong_simulated = Path(
        Path('.').parent, 'datasets', 'xiong_simulated').absolute()

    # Load URDF files
    GO1_URDF = NormalRobotGraph(path_to_go1_urdf, 'package://go1_description/',
                         'unitree_ros/robots/go1_description', True)
    model_type = 'gnn'

    # Initalize the datasets
    go1_sim_dataset = Go1SimulatedDataset(path_to_xiong_simulated, GO1_URDF,
                                          model_type)


def main():
    reader = AnyReader([
        Path(Path('.').parent,
             "datasets/xiong_simulated/raw/traj_0073.bag").absolute()
    ])
    reader.open()
    grf_gen = reader.messages(
        connections=[x for x in reader.connections if x.topic == "grf"])
    pos_gen = reader.messages(connections=[
        x for x in reader.connections if x.topic == "joint_positions"
    ])
    vel_gen = reader.messages(connections=[
        x for x in reader.connections if x.topic == "joint_velocities"
    ])
    tau_gen = reader.messages(connections=[
        x for x in reader.connections if x.topic == "joint_torques"
    ])

    # Extract all of the relevant data into arrays
    grf_data = []
    pos_data = []
    vel_data = []
    tau_data = []

    for connection, _timestamp, rawdata in grf_gen:
        msg = reader.deserialize(rawdata, connection.msgtype)
        grf_data.append(msg.data)
        print(type(msg.data[0]))
    for connection, _timestamp, rawdata in pos_gen:
        msg = reader.deserialize(rawdata, connection.msgtype)
        pos_data.append(msg.data)
    for connection, _timestamp, rawdata in vel_gen:
        msg = reader.deserialize(rawdata, connection.msgtype)
        vel_data.append(msg.data)
    for connection, _timestamp, rawdata in tau_gen:
        msg = reader.deserialize(rawdata, connection.msgtype)
        tau_data.append(msg.data)

    # Make sure we have the same amount of data in each array
    # Otherwise, we have an issue
    if len(grf_data) != len(pos_data) or len(pos_data) != len(vel_data) or \
        len(vel_data) != len(tau_data):
        raise ValueError("Dataset has different amounts of data.")

    index = 73
    for val in grf_data[index]:
        print(str(val) + ", ", end="")
    print()
    for val in pos_data[index]:
        print(str(val) + ", ", end="")
    print()
    for val in vel_data[index]:
        print(str(val) + ", ", end="")
    print()
    for val in tau_data[index]:
        print(str(val) + ", ", end="")
    print()


def simulate_process():
    bag_numbers = np.linspace(0, 99, 100)
    root = str(Path(Path('.').parent, 'datasets/xiong_simulated').absolute())
    processed_dir = str(
        Path(Path('.').parent,
             'datasets/xiong_simulated/processedTest').absolute())

    # Dataset has no sequence numbers, so hardcode first sequence number
    first_seq = 0
    curr_seq_num = first_seq
    print(bag_numbers)

    for val in bag_numbers:
        # Set up a reader to read the rosbag
        path_to_bag = os.path.join(
            root, 'raw', 'traj_' + str(int(val)).rjust(4, '0') + '.bag')
        reader = AnyReader([Path(path_to_bag)])
        reader.open()
        grf_gen = reader.messages(
            connections=[x for x in reader.connections if x.topic == "grf"])
        pos_gen = reader.messages(connections=[
            x for x in reader.connections if x.topic == "joint_positions"
        ])
        vel_gen = reader.messages(connections=[
            x for x in reader.connections if x.topic == "joint_velocities"
        ])
        tau_gen = reader.messages(connections=[
            x for x in reader.connections if x.topic == "joint_torques"
        ])

        # Extract all of the relevant data into arrays
        grf_data = []
        pos_data = []
        vel_data = []
        tau_data = []

        for connection, _timestamp, rawdata in grf_gen:
            msg = reader.deserialize(rawdata, connection.msgtype)
            grf_data.append(msg.data)
        for connection, _timestamp, rawdata in pos_gen:
            msg = reader.deserialize(rawdata, connection.msgtype)
            pos_data.append(msg.data)
        for connection, _timestamp, rawdata in vel_gen:
            msg = reader.deserialize(rawdata, connection.msgtype)
            vel_data.append(msg.data)
        for connection, _timestamp, rawdata in tau_gen:
            msg = reader.deserialize(rawdata, connection.msgtype)
            tau_data.append(msg.data)

        # Make sure we have the same amount of data in each array
        # Otherwise, we have an issue
        if len(grf_data) != len(pos_data) or len(pos_data) != len(vel_data) or \
            len(vel_data) != len(tau_data):
            raise ValueError("Dataset has different amounts of data.")

        # Iterate through the arrays and save important data in txt file
        for i in range(len(grf_data)):
            with open(str(Path(processed_dir,
                               str(curr_seq_num) + ".txt")), "w") as f:
                arrays = [grf_data[i], pos_data[i], vel_data[i], tau_data[i]]
                for array in arrays:
                    for val in array:
                        f.write(str(val) + " ")
                    f.write('\n')
                curr_seq_num += 1

        print(curr_seq_num)

    # Write a txt file to save the dataset length & and first sequence index
    length = curr_seq_num
    with open(os.path.join(processed_dir, "info.txt"), "w") as f:
        f.write(str(length) + " " + str(first_seq))


if __name__ == "__main__":
    main()
