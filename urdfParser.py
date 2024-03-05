from urchin import URDF
import os
import numpy as np
import networkx

class RobotURDF():
    def __init__(self, urdf_path):
        self.urdf_path = urdf_path
        self.update_paths_in_urdf()
        self.robot_urdf = URDF.load(self.new_urdf_path)

    def update_paths_in_urdf(self):
        ros_url = 'package://hyq_description/'
        actual_url = os.path.join(os.getcwd(), os.path.dirname(self.urdf_path)) + '/hyq-description/'
        self.new_urdf_path = self.urdf_path[:-5] + '_updated.urdf'

        # Load urdf file
        file_data = None
        with open(self.urdf_path, 'r') as f:
            file_data = f.readlines()

        # Replace all instances of ros_url with actual_url
        for i in range(0, len(file_data)):
            while ros_url in file_data[i]:
                file_data[i] = file_data[i].replace(ros_url, actual_url)

        # Save the updated urdf in a new location
        with open(self.new_urdf_path, 'w') as f:
            f.writelines(file_data)

    def get_edge_index(self):
        # Create a link dictionary to map link names to numbers
        link_names = []
        for link in self.robot_urdf.links:
            link_names.append(link.name)
        link_dict = dict(zip(link_names, range(len(self.robot_urdf.links))))

        # Iterate through joints, and add to the edge matrix
        edge_matrix = None
        for joint in self.robot_urdf.joints:
            a_index = link_dict[joint.parent]
            b_index = link_dict[joint.child]
            edge_vector = np.array([[a_index, b_index],
                                    [b_index, a_index]])
            if edge_matrix is None:
                edge_matrix = edge_vector
            else:
                edge_matrix = np.concatenate((edge_matrix, edge_vector), axis=1)
        
        return edge_matrix
    
    def display_URDF_info(self):
        print("============ Displaying Robot Links: ============")
        for link in self.robot_urdf.links:
            print(link.name)
        print("")

        print("============ Displaying Robot Joints: ============")
        for joint in self.robot_urdf.joints:
            print('{} -> {} <- {}'.format(joint.parent, joint.name, joint.child))
        print("")
            

def main():
    HyQ_URDF = RobotURDF('urdf_files/HyQ/hyq.urdf')
    HyQ_URDF.display_URDF_info()
    print("Edge Matrix: ", HyQ_URDF.get_edge_index())

if __name__ == "__main__":
    main()