from urchin import URDF
import os
import numpy as np
import urchin
from pathlib import Path

class InvalidURDFException(Exception):
    pass


class RobotGraph():

    class Node():
        """
        Simple class that holds a node's name, and the edges that
        may or may not connect them to a parent or a child.
        """

        def __init__(self, name: str, edge_parent: str,
                     edge_children: list[str], joint: urchin.Joint):
            self.name: str = name
            self.edge_parent: str = edge_parent
            self.edge_children: list[str] = edge_children
            self.joint = joint

        @staticmethod
        def get_list_of_node_types():
            """
            Returns a list of all the node types.
            """
            return ['base', 'joint', 'foot']

        def get_node_type(self):
            """
            Returns the name of the node type for a heterogenous graph,
            depending on the edges connected to this node.
            If it has a parent and a child, it's a 'joint' node. If it only 
            has a parent, it's a 'foot' node. If it only has children, it's a
            'base' node.
            """
            node_types = self.get_list_of_node_types()

            if self.edge_parent is not None and len(self.edge_children) > 0:
                return node_types[1]
            elif self.edge_parent is not None and len(self.edge_children) == 0:
                return node_types[2]
            elif self.edge_parent is None and len(self.edge_children) > 0:
                return node_types[0]
            else:
                raise Exception(
                    "Every node should have a child or parent edge.")

    class Edge():
        """
        Simple class that holds an edge's name, and
        the names of both connections.
        """

        def __init__(self, name: str, parent: str, child: str, link: urchin.Link):
            self.name = name
            self.parent = parent
            self.child = child
            self.link = link

    def __init__(self,
                 urdf_path: Path,
                 ros_builtin_path: str,
                 urdf_to_desc_path: str):
        """
        Constructor for RobotGraph class.

        Args:
            urdf_path (Path): The absolute path from this file (graphParser.py)
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
        self.urdf_path = str(urdf_path)
        self.new_urdf_path = self.urdf_path[:-5] + '_updated.urdf'
        self.ros_builtin_path = ros_builtin_path
        self.urdf_to_desc_path = urdf_to_desc_path

        # Make the updated urdf file, or replace it.
        # This avoids issues with updating URDF files,
        # but the new file not being updated.
        self.create_updated_urdf_file()

        # Load the URDF with updated paths
        self.robot_urdf = URDF.load(self.new_urdf_path)

        # Create edges from the links
        self.edges = []
        for link in self.robot_urdf.links:
            edge_parent, edge_children = self.get_connections_to_link(link)

            # If link doesn't have a parent and at least one child, drop it.
            # We can't make an edge out of it.
            if edge_parent is None and len(edge_children) == 0:
                raise InvalidURDFException("Link connected to no joints.")
            elif edge_parent is None or len(edge_children) == 0:
                continue

            # Create an edge from each link parent to each link child
            new_edge = None
            if (len(edge_children) == 1):
                self.edges.append(
                    self.Edge(name=link.name,
                                parent=edge_parent,
                                child=edge_children[0],
                                link=link))
            else:  # If there are multiple edges for this one link
                # give them each a unique name.
                for edge_child in edge_children:
                    self.edges.append(
                        self.Edge(name=link.name + "_to_" + edge_child,
                                    parent=edge_parent,
                                    child=edge_child,
                                    link=link))

        # Create nodes from the Joints
        self.nodes = []
        for joint in self.robot_urdf.joints:
            # Make sure the edge parent hasn't been pruned, and find
            # the updated name
            edge_parent = None
            for edge in self.edges:
                if joint.parent in edge.name and joint.name == edge.child:
                    edge_parent = edge.name

            # Make sure the edge child hasn't been pruned, and find
            # the updated name
            edge_child = []
            for edge in self.edges:
                if joint.child in edge.name and joint.name == edge.parent:
                    edge_child.append(edge.name)

            new_node = self.Node(name=joint.name,
                                    edge_parent=edge_parent,
                                    edge_children=edge_child,
                                    joint=joint)
            self.nodes.append(new_node)

    def get_connections_to_link(self, link):
        """
        This helper function finds any connections a link might have.

        Parameters:
            link: urchin.Link - The link to find connections to.
        
        Returns:
            edge_parent: str - The name of the parent edge, if it exists. None otherwise.
            edge_children: list[str] - A list of the names of the children edges. Could
                be an empty list if there are none.
        """
        # Get all the connections to this link
        connections = []
        for joint in self.robot_urdf.joints:
            # The joint's parent is the link's child, and visa versa
            if (joint.parent == link.name):
                connections.append((joint.name, 'child'))
            elif (joint.child == link.name):
                connections.append((joint.name, 'parent'))

        # Find the singular link parent
        parent_connection = None
        for conn in connections:
            if conn[1] == 'parent':
                if parent_connection is not None:
                    raise InvalidURDFException(
                        "Link has more than two parent joints.")
                else:
                    parent_connection = conn
                    connections.remove(conn)

        # Return the parent connection and all of the child actions
        edge_children = []
        for conn in connections:
            edge_children.append(conn[0])
        if parent_connection is None:
            edge_parent = None
        else:
            edge_parent = parent_connection[0]
        return edge_parent, edge_children

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
        corresponding index. Each node should have a unique
        index.

        Returns:
            (dict[str, int]): A dictionary that maps node name
                to index.
        """

        raise NotImplementedError

    def get_num_nodes(self):
        """
        Return the number of nodes in the graph.

        Returns:
            (int): Number of node nodes in graph.
        """
        return len(self.nodes)

    def get_node_from_name(self, node_name):
        """
        Helper function that returns a node based
        on the name provided. If no node with the
        name is found, None is returned.

        Parameters:
            node_name (str): Name of the node to return.
        """

        for node in self.nodes:
            if node.name == node_name:
                return node
        else:
            return None

    def display_graph_info(self):
        """
        Helper function that displays information about the robot
        graph on the command line in a readable format.
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


class NormalRobotGraph(RobotGraph):
    """
    Graph where all of the nodes are homogeneous.
    """

    def get_node_name_to_index_dict(self):
        node_names = []
        for node in self.nodes:
            node_names.append(node.name)
        return dict(zip(node_names, range(len(self.nodes))))

    def get_node_index_to_name_dict(self):
        """
        Return a dictionary that maps the node index to its
        name.

        Returns:
            (dict[int, str]): A dictionary that maps node index
                to name.
        """

        node_names = []
        for node in self.nodes:
            node_names.append(node.name)
        node_dict = dict(zip(range(len(self.nodes)), node_names))
        return node_dict

    def get_edge_index_matrix(self, heterogenous=False):
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
            edge_dict (dict[str, np.array)]): 
                This dictionary takes the name of an edge in the graph
                as input. It returns an np.array (N, 2), where N is the
                number of connections this name supports.
        """

        node_dict = self.get_node_name_to_index_dict()

        # Create a dictionary to map edge name pair to its node connections
        edge_dict = {}
        for edge in self.edges:
            edge_dict[edge.name] = np.array(
                [[node_dict[edge.parent], node_dict[edge.child]],
                 [node_dict[edge.child], node_dict[edge.parent]]])

        return edge_dict


class HeterogeneousRobotGraph(RobotGraph):
    """
    Heterogeneous graph, where the nodes are different types.
    """

    def _get_nodes_organized_by_type(self):
        """
        Organizes each node into a type with nodes of its same
        type.

        Returns:
            list[list[Node]]: The outer list holds as many lists
                as there are types, and each inner list holds
                all of the nodes of that particular type.
        """
        types = RobotGraph.Node.get_list_of_node_types()
        nodes = []
        for type in types:
            nodes_of_type = []
            for node in self.nodes:
                if node.get_node_type() == type:
                    nodes_of_type.append(node.name)
            nodes.append(nodes_of_type)
        return nodes

    def get_node_name_to_index_dict(self):
        """
        Because this graph is heterogenous, the nodes index
        depends of the type of the node. This also means that
        multiple nodes can have the same index.
        """

        all_lists = []
        for nodes_of_type in self._get_nodes_organized_by_type():
            all_lists = all_lists + list(
                zip(nodes_of_type, range(len(nodes_of_type))))

        return dict(all_lists)

    # TODO: Doesn't work currently
    # def get_node_index_to_name_dict(self):
    #     name_to_index = self.get_node_name_to_index_dict()
    #     return dict((i, n) for n, i in name_to_index.items())

    def get_num_of_each_node_type(self):
        """
        Returns the numbers of each node type.

        Returns:
            list[int]: List of number of each node type
        """
        nodes = self._get_nodes_organized_by_type()
        numbers = []
        for node_list in nodes:
            numbers.append(len(node_list))
        return numbers

    def get_edge_index_matrices(self):
        """
        Return the edge connectivity matrices.

        Returns:
            (list[np.array]): Multiple matrices, as outlined below:
                data['base', 'connect', 'joint'].edge_index -> [2, X]
                data['joint', 'connect', 'base'].edge_index -> [2, X]
                data['joint', 'connect', 'joint'].edge_index -> [2, Y]
                data['foot', 'connect', 'joint'].edge_index -> [2, Z]
                data['joint', 'connect', 'foot'].edge_index -> [2, Z]
        """

        def add_to_matrix(matrix, vector):
            """
            Helper function for appending to a numpy
            matrix along the 1st axis.
            """
            if matrix is None:
                matrix = vector
            else:
                matrix = np.concatenate((matrix, vector), axis=1)
            return matrix

        # Get the name to index dictionary
        node_dict = self.get_node_name_to_index_dict()

        # Define all of the edge matrices
        base_to_joint_matrix = None
        joint_to_joint_matrix = None
        foot_to_joint_matrix = None

        # Iterate through edges
        for edge in self.edges:

            # Get the nodes for the parent and the child
            parent_node: RobotGraph.Node = self.get_node_from_name(edge.parent)
            child_node: RobotGraph.Node = self.get_node_from_name(edge.child)

            # Get their types and indices
            parent_type = parent_node.get_node_type()
            child_type = child_node.get_node_type()
            p_index = node_dict[edge.parent]
            c_index = node_dict[edge.child]

            # Add their info to the corresponding matrix
            if parent_type == child_type and parent_type == 'joint':
                edge_vector = np.array([[p_index, c_index], [c_index,
                                                             p_index]])
                joint_to_joint_matrix = add_to_matrix(joint_to_joint_matrix,
                                                      edge_vector)
            elif parent_type == 'base' and child_type == 'joint':
                edge_vector = np.array([[p_index], [c_index]])
                base_to_joint_matrix = add_to_matrix(base_to_joint_matrix,
                                                     edge_vector)
            elif parent_type == 'joint' and child_type == 'foot':
                edge_vector = np.array([[c_index], [p_index]])
                foot_to_joint_matrix = add_to_matrix(foot_to_joint_matrix,
                                                     edge_vector)
            else:
                raise Exception("Not possible")

        # Create the last two matrices
        joint_to_base_matrix = base_to_joint_matrix[[1, 0]]
        joint_to_foot_matrix = foot_to_joint_matrix[[1, 0]]

        return base_to_joint_matrix, joint_to_base_matrix, joint_to_joint_matrix, \
               foot_to_joint_matrix, joint_to_foot_matrix
