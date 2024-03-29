from pathlib import Path
import unittest
import os
from grfgnn import RobotURDF
import pandas as pd
import numpy as np
from pathlib import Path


class TestRobotURDF(unittest.TestCase):

    def setUp(self):
        self.hyq_path = Path(
            Path(__file__).cwd(), 'urdf_files', 'HyQ', 'hyq.urdf').absolute()

        self.HyQ_URDF = RobotURDF(self.hyq_path, 'package://hyq_description/',
                                  'hyq-description', False)
        self.HyQ_URDF_swapped = RobotURDF(self.hyq_path,
                                          'package://hyq_description/',
                                          'hyq-description', True)

    def test_constructor(self):
        """
        Check if self.nodes has all the name in the URDF file, and
        check if self.edges has all the name in the URDF file.
        """

        nodes_name = {
            'world', 'base_link', 'trunk', 'lf_hipassembly', 'lh_hipassembly',
            'rf_hipassembly', 'rh_hipassembly', 'lf_upperleg', 'lh_upperleg',
            'rf_upperleg', 'rh_upperleg', 'lf_lowerleg', 'lh_lowerleg',
            'rf_lowerleg', 'rh_lowerleg', 'lf_foot', 'lh_foot', 'rf_foot',
            'rh_foot'
        }

        edges_name = {
            'floating_base_joint', 'floating_base', 'lf_haa_joint',
            'lh_haa_joint', 'rf_haa_joint', 'rh_haa_joint', 'lf_hfe_joint',
            'lh_hfe_joint', 'rf_hfe_joint', 'rh_hfe_joint', 'lf_kfe_joint',
            'lh_kfe_joint', 'rf_kfe_joint', 'rh_kfe_joint', 'lf_foot_joint',
            'lh_foot_joint', 'rf_foot_joint', 'rh_foot_joint'
        }

        for node in self.HyQ_URDF.nodes:
            self.assertTrue(node.name in nodes_name)

        for edge in self.HyQ_URDF.edges:
            self.assertTrue(edge.name in edges_name)

    def test_create_updated_urdf_file(self):
        """
        Check that calling the constructor creates
        the updated urdf file.
        """

        # Delete the urdf file
        hyq_path_updated = self.hyq_path.parent / "hyq_updated.urdf"
        os.remove(str(hyq_path_updated))
        self.assertFalse(os.path.exists(hyq_path_updated))

        # Rebuild it
        RobotURDF(self.hyq_path, 'package://hyq_description/',
                  'hyq-description', False)
        self.assertTrue(os.path.exists(hyq_path_updated))

    def test_get_node_name_to_index_dict(self):
        """
        Check if all the indexes of the nodes in the dictionary
        are unique.
        """

        key = list(self.HyQ_URDF.get_node_name_to_index_dict())
        get_nodes_index = []

        for key in key:
            index = self.HyQ_URDF.get_node_name_to_index_dict()[key]
            get_nodes_index.append(index)

        self.assertTrue(pd.Index(get_nodes_index).is_unique)

    def test_get_node_index_to_name_dict(self):
        """
        Check the index_to_name dict by running making sure the
        index_to_name dict and the name_to_index dict are consistent.
        """

        index_to_name = list(self.HyQ_URDF.get_node_index_to_name_dict())
        name_to_index = list(self.HyQ_URDF.get_node_name_to_index_dict())
        get_nodes_index = []

        for key in name_to_index:
            index = self.HyQ_URDF.get_node_name_to_index_dict()[key]
            get_nodes_index.append(index)

        self.assertEqual(index_to_name, get_nodes_index)

    def test_get_edge_index_matrix(self):
        """
        Check the dimensionality of the edge matrix.
        """

        edge_matrix = self.HyQ_URDF.get_edge_index_matrix()

        self.assertEqual(edge_matrix.shape[0], 2)
        self.assertEqual(edge_matrix.shape[1], 36)

    def test_get_num_nodes(self):
        """
        Check that the number of nodes are correct. 
        """

        self.assertEqual(self.HyQ_URDF.get_num_nodes(), 19)
        self.assertEqual(self.HyQ_URDF_swapped.get_num_nodes(), 18)

    def test_get_edge_connections_to_name_dict(self):
        """
        Check the connections_to_name dict by running making sure the
        connections_to_name dict and the name_to_connections dict are 
        consistent.
        """

        connections_to_name = list(
            self.HyQ_URDF.get_edge_connections_to_name_dict())
        name_to_connections = list(
            self.HyQ_URDF.get_edge_name_to_connections_dict())

        result = []
        for key in name_to_connections:
            connections = self.HyQ_URDF.get_edge_name_to_connections_dict(
            )[key]
            for i in range(connections.shape[1]):
                real_reshaped = np.squeeze(connections[:, i].reshape(1, -1))
                result.append(real_reshaped)

        result = [tuple(arr) for arr in result]

        self.assertEqual(connections_to_name, result)

    def test_get_edge_name_to_connections_dict(self):
        """
        Check each connection in the dictionary is unique.
        """

        name_to_connections = list(
            self.HyQ_URDF.get_edge_name_to_connections_dict())
        all_connections = []

        # Get all connections from dictionary
        for key in name_to_connections:
            connections = self.HyQ_URDF.get_edge_name_to_connections_dict(
            )[key]
            for i in range(connections.shape[1]):
                real_reshaped = np.squeeze(connections[:, i].reshape(1, -1))
                all_connections.append(real_reshaped)

        seen_arrays = set()
        for array in all_connections:
            # Convert the array to a tuple since lists are not hashable
            array_tuple = tuple(array)

            # Make sure the array hasn't been seen
            self.assertTrue(array_tuple not in seen_arrays)

            # Add it to the seen arrays
            seen_arrays.add(array_tuple)


if __name__ == '__main__':
    unittest.main()
