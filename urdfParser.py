from urchin import URDF
import os
import numpy as np


class RobotURDF():

    def __init__(self, urdf_path, ros_builtin_path, urdf_to_desc_path):
        """
        Constructor for RobotURDF class.

        Args:
            urdf_path (str): The relative path from this file (urdfParser.py)
                to the desired urdf file to load.
            ros_builtin_path (str): The path ROS uses in the urdf file to navigate
                to the urdf description directory. An example looks like this:
                "package://a1_description/". You can find this by manually looking
                in the urdf file.
            urdf_to_desc_path (str): The relative path from the urdf file to
                the urdf description directory. This directory typically contains
                folders like "meshes" and "urdf".
        """

        # Define paths
        self.urdf_path = urdf_path
        self.new_urdf_path = self.urdf_path[:-5] + '_updated.urdf'
        self.ros_builtin_path = ros_builtin_path
        self.urdf_to_desc_path = urdf_to_desc_path

        # Regenerate the updated urdf if it isn't already made
        if not os.path.isfile(self.new_urdf_path):
            self.create_updated_urdf_file()

        # Load the URDF with updated paths
        self.robot_urdf = URDF.load(self.new_urdf_path)

    def create_updated_urdf_file(self):
        """
        This method takes the given urdf file and creates a new one
        that replaces the ROS paths (to the urdf description repo) 
        with the current system paths. Now, the "URDF.load()" function 
        will run properly.
        """

        # Calculate the actual path to the urdf description repository
        actual_url = os.path.join(os.getcwd(), os.path.dirname(self.urdf_path),
                                  self.urdf_to_desc_path, "temp")[:-4]

        # Load urdf file
        file_data = None
        with open(self.urdf_path, 'r') as f:
            file_data = f.readlines()

        # Replace all instances of ros_builtin_path with the actual path
        # for the current system
        for i in range(0, len(file_data)):
            while self.ros_builtin_path in file_data[i]:
                file_data[i] = file_data[i].replace(self.ros_builtin_path,
                                                    actual_url)

        # Save the updated urdf in a new location
        with open(self.new_urdf_path, 'w') as f:
            f.writelines(file_data)

    def get_link_name_to_index_dict(self):
        """
        Return a dictionary that maps the link name to its
        corresponding index.

        Returns:
            (dict[str, int]): A dictinoary that maps link name
                to index.
        """
        link_names = []
        for link in self.robot_urdf.links:
            link_names.append(link.name)
        return dict(zip(link_names, range(len(self.robot_urdf.links))))

    def get_link_index_to_name_dict(self):
        """
        Return a dictionary that maps the link index to its
        name

        Returns:
            (dict[int, str]): A dictionary that maps link index
                to name.
        """

        link_names = []
        for link in self.robot_urdf.links:
            link_names.append(link.name)
        link_dict = dict(zip(range(len(self.robot_urdf.links)), link_names))
        return link_dict

    def get_edge_index_matrix(self):
        """
        Return the edge connectivity matrix, which defines each edge connection
        to each joint. This matrix is the 'edge_index' matrix passed to the PyTorch
        Geometric Data class: https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.data.Data.html#torch_geometric.data.Data . However, it will need to be converted to a torch.Tensor first.

        Returns:
            edge_index (np.array): A 2xN matrix, where N is the twice the number of
                edge connections. This is because each connection must 
                be put into the matrix twice, as each edge is bidirectional
                For example, a connection between links 0 and 1 will be
                found in the edge_index matrix as [[0, 1], [1, 0]].

        """
        link_dict = self.get_link_name_to_index_dict()

        # Iterate through joints, and add to the edge matrix
        edge_matrix = None
        for joint in self.robot_urdf.joints:
            a_index = link_dict[joint.parent]
            b_index = link_dict[joint.child]
            edge_vector = np.array([[a_index, b_index], [b_index, a_index]])
            if edge_matrix is None:
                edge_matrix = edge_vector
            else:
                edge_matrix = np.concatenate((edge_matrix, edge_vector),
                                             axis=1)

        return edge_matrix

    def get_num_links(self):
        """
        Return the number of links in the URDF file.

        Returns:
            (int): Number of link nodes in URDF.
        """
        return len(self.robot_urdf.links)

    def get_edge_connections_to_name_dict(self):
        """
        Return a dictionary that maps a tuple of two link indices
        and return the name of the edge that connects to both of those 
        links.

        Returns:
            joint_dict (dict[tuple(int, int), str]): This dictionary
                takes a tuple of two link indices. It 
                returns the name of the edge that connects them.
        """

        link_dict = self.get_link_name_to_num_dict()

        # Create a dictionary to map edge pair to a joint name
        joint_dict = {}
        for joint in self.robot_urdf.joints:
            joint_dict[(link_dict[joint.parent],
                        link_dict[joint.child])] = joint.name
            joint_dict[(link_dict[joint.child],
                        link_dict[joint.parent])] = joint.name

        return joint_dict

    def get_edge_name_to_connections_dict(self):
        """
        Return a dictionary that maps the edge name to the links
        it's connected to.

        Returns:
            joint_dict (dict[str, np.array([int, int])]): This dictionary
                takes the name of an edge in the URDF as input. It 
                returns an np.array that contains the indices of the 
                two links that this edge is connected to.
        """

        link_dict = self.get_link_name_to_index_dict()

        # Create a dictionary to map edge pair to a joint name
        joint_dict = {}
        for joint in self.robot_urdf.joints:
            joint_dict[joint.name] = np.array(
                [link_dict[joint.parent], link_dict[joint.child]])
        return joint_dict

    def display_URDF_info(self):
        """
        Helper function that displays information about the robot
        URDF on the command line in a readable format.
        """

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
            print('{} -> {} <- {}'.format(joint.parent, joint.name,
                                          joint.child))
        print("")


def main():
    """
    Simple code that demonstrates basic functionality of the RobotURDF class.
    """

    HyQ_URDF = RobotURDF('urdf_files/HyQ/hyq.urdf',
                         'package://hyq_description/', 'hyq-description')
    HyQ_URDF.display_URDF_info()
    print("Edge Matrix (HyQ): ", HyQ_URDF.get_edge_index_matrix())

    A1_URDF = RobotURDF('urdf_files/A1/a1.urdf', 'package://a1_description/',
                        'unitree_ros/robots/a1_description')
    A1_URDF.display_URDF_info()
    print("Edge Matrix (A1): ", A1_URDF.get_edge_index_matrix())


if __name__ == "__main__":
    main()
