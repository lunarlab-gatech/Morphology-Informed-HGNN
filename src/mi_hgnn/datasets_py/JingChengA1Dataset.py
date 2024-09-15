import rosbag
import csv
import os
from std_msgs.msg import Int8, Float32MultiArray
from unitree_legged_msgs.msg import IMU, LowCmd, LowState
from rosgraph_msgs.msg import Log

class JingChengA1Dataset:
    def __init__(self, bag_file_path, output_dir='dataset_output'):
        """
        Initialize the dataset class with the ROSbag file path and output directory.
        """
        self.bag_file_path = bag_file_path
        self.output_dir = output_dir
        
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

    def extract_data(self):
        """
        Extract data from the ROSbag file and store it into CSV format.
        """
        with rosbag.Bag(self.bag_file_path, 'r') as bag:
            # Create dictionaries for storing data by topic
            jump_switch_data = []
            low_cmd_data = []
            low_state_desired_data = []
            imu_data = []
            low_state_data = []
            rosout_data = []

            # Read through all messages in the ROSbag
            for topic, msg, t in bag.read_messages():
                if topic == '/3_2/jump_switch':
                    jump_switch_data.append(self.process_int8(msg))
                elif topic == '/3_2/lowCmd/cmd':
                    low_cmd_data.append(self.process_low_cmd(msg))
                elif topic == '/3_2/lowState/desired':
                    low_state_desired_data.append(self.process_float32_multi_array(msg))
                elif topic == '/3_2/lowState/imu':
                    imu_data.append(self.process_imu(msg))
                elif topic == '/3_2/lowState/state':
                    low_state_data.append(self.process_low_state(msg))
                elif topic in ['/rosout', '/rosout_agg']:
                    rosout_data.append(self.process_log(msg))
                else:
                    print(f"Skipping unknown topic: {topic}")

            # Save the data to CSV files
            self.save_to_csv(jump_switch_data, 'jump_switch.csv', ['time', 'data'])
            self.save_to_csv(low_cmd_data, 'low_cmd.csv', ['time', 'cmd_data'])
            self.save_to_csv(low_state_desired_data, 'low_state_desired.csv', ['time', 'data'])
            self.save_to_csv(imu_data, 'imu_data.csv', ['time', 'orientation_w', 'orientation_x', 'orientation_y', 'orientation_z',
                                                        'angular_velocity_x', 'angular_velocity_y', 'angular_velocity_z',
                                                        'linear_acceleration_x', 'linear_acceleration_y', 'linear_acceleration_z'])
            self.save_to_csv(low_state_data, 'low_state.csv', ['time', 'q', 'qd', 'tau', 'footForce', 'footForceEst', 'imu_acc', 'imu_gyro'])
            self.save_to_csv(rosout_data, 'rosout.csv', ['time', 'message'])

    def process_int8(self, msg):
        """
        Process and format Int8 messages.
        """
        return {'time': msg.header.stamp.to_sec(), 'data': msg.data}

    def process_float32_multi_array(self, msg):
        """
        Process and format Float32MultiArray messages.
        """
        return {'time': msg.layout.data_offset, 'data': msg.data}

    def process_imu(self, msg):
        """
        Process and format IMU messages from unitree_legged_msgs/IMU.
        """
        return {
            'time': msg.header.stamp.to_sec(),
            'orientation_w': msg.orientation.w,
            'orientation_x': msg.orientation.x,
            'orientation_y': msg.orientation.y,
            'orientation_z': msg.orientation.z,
            'angular_velocity_x': msg.angular_velocity.x,
            'angular_velocity_y': msg.angular_velocity.y,
            'angular_velocity_z': msg.angular_velocity.z,
            'linear_acceleration_x': msg.linear_acceleration.x,
            'linear_acceleration_y': msg.linear_acceleration.y,
            'linear_acceleration_z': msg.linear_acceleration.z
        }

    def process_low_cmd(self, msg):
        """
        Process and format LowCmd messages from unitree_legged_msgs/LowCmd.
        """
        return {
            'time': msg.header.stamp.to_sec(),
            'cmd_data': msg
        }

    def process_low_state(self, msg):
        """
        Process and format LowState messages from unitree_legged_msgs/LowState.
        """
        return {
            'time': msg.header.stamp.to_sec(),
            'q': msg.q,  # Joint positions
            'qd': msg.qd,  # Joint velocities
            'tau': msg.tau,  # Joint torques
            'footForce': msg.footForce,  # Force on foot
            'footForceEst': msg.footForceEst,  # Estimated foot force
            'imu_acc': msg.imu.acceleration,
            'imu_gyro': msg.imu.gyroscope
        }

    def process_log(self, msg):
        """
        Process and format Log messages from rosgraph_msgs/Log.
        """
        return {'time': msg.header.stamp.to_sec(), 'message': msg.msg}

    def save_to_csv(self, data, filename, fieldnames):
        """
        Save the extracted data to a CSV file.
        """
        file_path = os.path.join(self.output_dir, filename)
        with open(file_path, 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            for row in data:
                writer.writerow(row)

# Usage
dataset = JingChengA1Dataset('/home/lunarlab/Downloads/rosbag_test/2023-07-10-14-08-41.bag')
dataset.extract_data()
