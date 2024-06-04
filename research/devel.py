from grfgnn.graphParser import NormalRobotGraph
from pathlib import Path
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


def get_regression_tests():
     # Set up a reader to read the rosbag
    root = "/home/dlittleman/state-estimation-gnn/datasets/QuadSDK-A1Speed1.0"
    path_to_bag = os.path.join(
        root, 'raw', 'data.bag')
    reader = AnyReader([Path(path_to_bag)])
    reader.open()
    connections = [x for x in reader.connections if x.topic == '/quadruped_dataset_entries']

    # Iterate through the generators and write important data
    # to a file
    prev_grf_time, prev_joint_time, prev_imu_time = 0, 0, 0
    dataset_entries = 0
    index = 0
    for connection, _timestamp, rawdata in reader.messages(connections=connections):

            data = reader.deserialize(rawdata, connection.msgtype)
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
            
            arrays = []

            # Add on the timestamp info
            if index >= 9999 and index <= 10001:
                print("index: ", index)
                print("Timestamps: ", str([grf_time, joint_time, imu_time]))

                # Add on the GRF data
                grf_vec = grf_data.vectors
                print("GRFs: ", str([grf_vec[0].x, grf_vec[0].y, grf_vec[0].z,
                            grf_vec[1].x, grf_vec[1].y, grf_vec[1].z,
                            grf_vec[2].x, grf_vec[2].y, grf_vec[2].z,
                            grf_vec[3].x, grf_vec[3].y, grf_vec[3].z]))

                # Add on the IMU data
                print("LA: ", str([
                    imu_data.linear_acceleration.x,
                    imu_data.linear_acceleration.y,
                    imu_data.linear_acceleration.z
                ]))
                print("AV: ", str([
                    imu_data.angular_velocity.x, 
                    imu_data.angular_velocity.y,
                    imu_data.angular_velocity.z
                ]))

                # Add on the joint data
                print("Joint Position: ", str(joint_data.joints.position))
                print("Joint Velocity: ", str(joint_data.joints.velocity))
                print("Joint Effort: ", str(joint_data.joints.effort))

            index = index + 1
    
    curr = [-1.01236671,0.4598878, 0.15727989,1.94620645,1.37242715,3.65867992,-1.26258874,2.21378912,3.51958775,1.03195194, -0.34262034,0.32344725]
    twoback = [-1.02946559,0.46104683,0.16103043,1.92477756,1.28487106,3.62581783,
 -1.32279199,2.24824336,3.46041838,0.99728071, -0.35475192,0.31078379]
    
    for i in range(0, 12):
        print((curr[i]-twoback[i])/0.005)
    

if __name__ == "__main__":
  get_regression_tests()
