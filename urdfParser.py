from urchin import URDF
import os
import numpy as np

class RobotURDF():
    def __init__(self, urdf_path):
        self.urdf_path = urdf_path
        self.update_paths_in_urdf()
        self.robot_urdf = URDF.load(self.new_urdf_path)

    def update_paths_in_urdf(self):
        ros_url = 'package://hyq_description/'
        actual_url = os.path.join(os.getcwd(), os.path.dirname(self.urdf_path), "hyq-description", "temp")[:-4]
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

    # Create a link dictionary to map link names to numbers
    def get_link_name_to_num_dict(self):
        link_names = []
        for link in self.robot_urdf.links:
            link_names.append(link.name)
        return dict(zip(link_names, range(len(self.robot_urdf.links))))

    def get_edge_index(self):
        link_dict = self.get_link_name_to_num_dict()

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
    
    def get_num_nodes(self):
        return len(self.robot_urdf.links)
    
    def get_node_name_dict(self):
        node_names = []
        for link in self.robot_urdf.links:
            node_names.append(link.name)
        node_dict = dict(zip(range(len(self.robot_urdf.links)), node_names))
        return node_dict
    
    def get_edge_name_dict(self):
        link_dict = self.get_link_name_to_num_dict()

        # Create a dictionary to map edge pair to a joint name
        joint_dict = {}
        for joint in self.robot_urdf.joints:
            joint_dict[(link_dict[joint.parent], link_dict[joint.child])] = joint.name
        return joint_dict
    
    def display_URDF_info(self):
        print("============ Displaying Robot Links: ============")
        print("Link name: Mass - Inertia")
        for link in self.robot_urdf.links:
            print(f"{link.name}")
        print("")

        print("============ Displaying Example Link: ============")
        ex_link = self.robot_urdf.links[7]
        print("Name: ", ex_link.name)
        print("Mass: ", ex_link.inertial.mass)
        print("Inertia: \n", ex_link.inertial.inertia)
        print("Origin of Inertials: \n", ex_link.inertial.origin)
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