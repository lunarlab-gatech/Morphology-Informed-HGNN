from urchin import URDF
import urchin
import os
import numpy as np


class RobotURDF():

    class Node():
        """
        Simple class that holds a node's name.
        """

        def __init__(self, name):
            self.name = name

    class Edge():
        """
        Simple class that holds an edge's name, and
        both connections.
        """

        def __init__(self, name, parent, child):
            self.name = name
            self.parent = parent
            self.child = child

    def __init__(self,
                 urdf_path,
                 ros_builtin_path,
                 urdf_to_desc_path,
                 swap_nodes_and_edges=False):
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
            swap_nodes_and_edges (bool): If False, links in the URDF will be represented
                as nodes, and joints will be represented as edges. If True, links will
                be represented as edges, and joints will be represented as nodes. This
                will drop links that aren't connected to two joints. Default=False.
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

        # Setup nodes and edges
        if not swap_nodes_and_edges:
            self.nodes = []
            for link in self.robot_urdf.links:
                self.nodes.append(self.Node(name=link.name))
            self.edges = []
            for joint in self.robot_urdf.joints:
                self.edges.append(
                    self.Edge(name=joint.name,
                              parent=joint.parent,
                              child=joint.child))
        else:
            # Create Links from the Joints, to set as the nodes
            self.nodes = []
            for joint in self.robot_urdf.joints:
                new_node = self.Node(name=joint.name)
                self.nodes.append(new_node)

            # Create joints from the links, to set as the edges
            self.edges = []
            for link in self.robot_urdf.links:
                # Check that two joints connect to this link
                connections = []
                for joint in self.robot_urdf.joints:
                    if (joint.parent == link.name or joint.child == link.name):
                        connections.append(joint.name)

                # If two joints do connect, add it as an edge
                if len(connections) == 2:
                    new_edge = self.Edge(name=link.name,
                                         parent=connections[0],
                                         child=connections[1])
                    self.edges.append(new_edge)
                if len(connections) > 2:
                    # Setup an edge for each pair of connections
                    count = 0
                    for i in range(0, len(connections)):
                        for j in range(i + 1, len(connections)):
                            self.edges.append(
                                self.Edge(name=link.name + str(count),
                                          parent=connections[i],
                                          child=connections[j]))

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

    def get_node_name_to_index_dict(self):
        """
        Return a dictionary that maps the node name to its
        corresponding index.

        Returns:
            (dict[str, int]): A dictinoary that maps node name
                to index.
        """
        node_names = []
        for node in self.nodes:
            node_names.append(node.name)
        return dict(zip(node_names, range(len(self.nodes))))

    def get_node_index_to_name_dict(self):
        """
        Return a dictionary that maps the node index to its
        name

        Returns:
            (dict[int, str]): A dictionary that maps node index
                to name.
        """

        node_names = []
        for node in self.nodes:
            node_names.append(node.name)
        node_dict = dict(zip(range(len(self.nodes)), node_names))
        return node_dict

    def get_edge_index_matrix(self):
        """
        Return the edge connectivity matrix, which defines each edge connection
        to each node. This matrix is the 'edge_index' matrix passed to the PyTorch
        Geometric Data class: https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.data.Data.html#torch_geometric.data.Data . However, it will need to be converted to a torch.Tensor first.

        Returns:
            edge_index (np.array): A 2xN matrix, where N is the twice the number of
                edge connections. This is because each connection must 
                be put into the matrix twice, as each edge is bidirectional
                For example, a connection between nodes 0 and 1 will be
                found in the edge_index matrix as [[0, 1], [1, 0]].

        """
        node_dict = self.get_node_name_to_index_dict()

        # Iterate through edges, and add to the edge matrix
        edge_matrix = None
        for edge in self.edges:
            a_index = node_dict[edge.parent]
            b_index = node_dict[edge.child]
            edge_vector = np.array([[a_index, b_index], [b_index, a_index]])
            if edge_matrix is None:
                edge_matrix = edge_vector
            else:
                edge_matrix = np.concatenate((edge_matrix, edge_vector),
                                             axis=1)

        return edge_matrix

    def get_num_nodes(self):
        """
        Return the number of nodes in the URDF file.

        Returns:
            (int): Number of node nodes in URDF.
        """
        return len(self.nodes)

    def get_edge_connections_to_name_dict(self):
        """
        Return a dictionary that maps a tuple of two node indices
        and return the name of the edge that connects to both of those 
        nodes.

        Returns:
            edge_dict (dict[tuple(int, int), str]): This dictionary
                takes a tuple of two node indices. It 
                returns the name of the edge that connects them.
        """

        node_dict = self.get_node_name_to_index_dict()

        # Create a dictionary to map edge pair to an edge name
        edge_dict = {}
        for edge in self.edges:
            edge_dict[(node_dict[edge.parent],
                       node_dict[edge.child])] = edge.name
            edge_dict[(node_dict[edge.child],
                       node_dict[edge.parent])] = edge.name

        return edge_dict

    def get_edge_name_to_connections_dict(self):
        """
        Return a dictionary that maps the edge name to the nodes
        it's connected to.

        Returns:
            edge_dict (dict[str, np.array([int, int])]): This dictionary
                takes the name of an edge in the URDF as input. It 
                returns an np.array that contains the indices of the 
                two nodes that this edge is connected to.
        """

        node_dict = self.get_node_name_to_index_dict()

        # Create a dictionary to map edge name pair to its node connections
        edge_dict = {}
        for edge in self.edges:
            edge_dict[edge.name] = np.array([
                [node_dict[edge.parent], node_dict[edge.child]],
                [node_dict[edge.child], node_dict[edge.parent]]
                ])
            
        # print(edge_dict)
        return edge_dict

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

    HyQ_URDF = RobotURDF('urdf_files\HyQ\hyq.urdf',
                         'package://hyq_description/', 'hyq-description')
    # HyQ_URDF.display_URDF_info()
    # print("Edge Matrix (HyQ): ", HyQ_URDF.get_node_name_to_index_dict())

    # A1_URDF = RobotURDF('urdf_files\A1\a1.urdf', 'package://a1_description/',
    #                     'unitree_ros/robots/a1_description', True)
    # A1_URDF.display_URDF_info()
    # print("Edge Matrix (A1): ", A1_URDF.get_edge_index_matrix())

    HyQ_URDF.get_edge_name_to_connections_dict()


if __name__ == "__main__":
    main()
