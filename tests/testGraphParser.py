from pathlib import Path
import unittest
from mi_hgnn import RobotGraph, NormalRobotGraph, HeterogeneousRobotGraph
from pathlib import Path
import copy
import pandas as pd
import numpy as np
import os
import urchin
import numpy


class TestNormalRobotGraph(unittest.TestCase):

    def setUp(self):
        self.mini_cheetah_path = Path(
            Path(__file__).cwd(), 'urdf_files', 'MiniCheetah', 'miniCheetah.urdf').absolute()

        self.mini_cheetah_URDF = NormalRobotGraph(self.mini_cheetah_path,
                'package://yobotics_description/', 'mini-cheetah-gazebo-urdf/yobo_model/yobotics_description')

    def test_constructor(self):
        """
        Check that the constructor properly assigns all of the links and joints to a node/edge.
        """

        joint_names = [
            'floating_base', 
            'FL_hip_joint', 'FL_thigh_joint', 'FL_calf_joint', 'FL_foot_fixed', 
            'FR_hip_joint', 'FR_thigh_joint', 'FR_calf_joint', 'FR_foot_fixed', 
            'RL_hip_joint', 'RL_thigh_joint', 'RL_calf_joint', 'RL_foot_fixed', 
            'RR_hip_joint', 'RR_thigh_joint', 'RR_calf_joint', 'RR_foot_fixed', 
        ]

        edge_names_copy = copy.deepcopy(joint_names)
        for i, node in enumerate(self.mini_cheetah_URDF.nodes):
            self.assertTrue(node.name in edge_names_copy)
            edge_names_copy.remove(node.name)
        self.assertEqual(0, len(edge_names_copy))

        # Remember that links that don't connect to two or more
        # joints get dropped, as they can't be represented as an edge.
        # Additionally, links with multiple children joints get one
        # edge for each child.
        desired_edges = [
            RobotGraph.Edge('trunk_to_FL_hip_joint', "floating_base",
                            "FL_hip_joint", None),
            RobotGraph.Edge('trunk_to_FR_hip_joint', "floating_base",
                            "FR_hip_joint", None),
            RobotGraph.Edge('trunk_to_RL_hip_joint', "floating_base",
                            "RL_hip_joint", None),
            RobotGraph.Edge('trunk_to_RR_hip_joint', "floating_base",
                            "RR_hip_joint", None),
            RobotGraph.Edge('FL_hip', "FL_hip_joint", "FL_thigh_joint", None),
            RobotGraph.Edge('FL_thigh', "FL_thigh_joint", "FL_calf_joint", None),
            RobotGraph.Edge('FL_calf', "FL_calf_joint", "FL_foot_fixed", None),
            RobotGraph.Edge('FR_hip', "FR_hip_joint", "FR_thigh_joint", None),
            RobotGraph.Edge('FR_thigh', "FR_thigh_joint", "FR_calf_joint", None),
            RobotGraph.Edge('FR_calf', "FR_calf_joint", "FR_foot_fixed", None),
            RobotGraph.Edge('RL_hip', "RL_hip_joint", "RL_thigh_joint", None),
            RobotGraph.Edge('RL_thigh', "RL_thigh_joint", "RL_calf_joint", None),
            RobotGraph.Edge('RL_calf', "RL_calf_joint", "RL_foot_fixed", None),
            RobotGraph.Edge('RR_hip', "RR_hip_joint", "RR_thigh_joint", None),
            RobotGraph.Edge('RR_thigh', "RR_thigh_joint", "RR_calf_joint", None),
            RobotGraph.Edge('RR_calf', "RR_calf_joint", "RR_foot_fixed", None)
        ]
        for i, edge in enumerate(self.mini_cheetah_URDF.edges):
            match_found = False
            for j, desired_edge in enumerate(desired_edges):
                if edge.name == desired_edge.name:
                    self.assertEqual(edge.child, desired_edge.child)
                    self.assertEqual(edge.parent, desired_edge.parent)
                    desired_edges.remove(desired_edge)
                    match_found = True
                    break
            self.assertTrue(match_found)
        self.assertEqual(0, len(desired_edges))

        # ==================
        # Check that the nodes are given appropriate
        # labels based on their position in the graph.
        # ==================

        des_node_type = [
            'base', 'joint', 'joint', 'joint', 'foot', 'joint', 'joint',
            'joint', 'foot', 'joint', 'joint', 'joint', 'foot', 'joint',
            'joint', 'joint', 'foot'
        ]
        edge_names_copy = copy.deepcopy(joint_names)
        num_matches = 0
        for i, node in enumerate(self.mini_cheetah_URDF.nodes):
            for j, node_des in enumerate(edge_names_copy):
                if (node.name == node_des):
                    self.assertEqual(node.get_node_type(), des_node_type[j])
                    num_matches += 1
                    break
        self.assertEqual(num_matches, len(des_node_type))

        # ==================
        # Check that the proper Link information is stored for 
        # one of the edges, and that the proper Joint information
        # is stored for one of the Nodes.
        # ==================
        node_found = False
        for i, node in enumerate(self.mini_cheetah_URDF.nodes):
            if node.name == "RL_hip_joint":
                # Test name information
                joint: urchin.Joint= node.joint
                self.assertEqual("RL_hip_joint", joint.name)

                # Test joint information
                np.testing.assert_array_equal(np.array([[1.0, 0.0, 0.0, -0.196], [0.0, 1.0, 0.0, 0.049664], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]]), joint.origin)
                self.assertEqual(0.0, joint.dynamics.damping)
                self.assertEqual(0.0, joint.dynamics.friction)

                node_found = True
                break

        self.assertTrue(node_found)
        
        edge_found = False
        for i, edge in enumerate(self.mini_cheetah_URDF.edges):
            if edge.name == "FL_calf":
                # Test name information
                link: urchin.Link = edge.link
                self.assertEqual("FL_calf", link.name)

                # Test inertial information
                np.testing.assert_array_equal(np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]]), link.inertial.origin)
                self.assertEqual(0.064, link.inertial.mass)
                np.testing.assert_array_equal(np.array([[0.000214698666667, 0.0, 0.0], [0.0, 0.000214698666667, 0.0], [0.0, 0.0, 2.73066666667e-06]]), link.inertial.inertia)

                edge_found = True
                break

        self.assertTrue(edge_found)
        
    def test_get_connections_to_link(self):
        """
        Check that we can properly find the connections to the links in the library.
        """

        edge_parent, edge_children = self.mini_cheetah_URDF.get_connections_to_link(
            urchin.Link("base_link", None, None, None))
        self.assertEqual(edge_parent, None)
        self.assertSequenceEqual(edge_children, ["floating_base"])

        edge_parent, edge_children = self.mini_cheetah_URDF.get_connections_to_link(
            urchin.Link("trunk", None, None, None))
        self.assertEqual(edge_parent, "floating_base")
        self.assertSequenceEqual(
            edge_children,
            ["RL_hip_joint", "FL_hip_joint", "RR_hip_joint", "FR_hip_joint"])

        edge_parent, edge_children = self.mini_cheetah_URDF.get_connections_to_link(
            urchin.Link("FL_foot", None, None, None))
        self.assertEqual(edge_parent, "FL_foot_fixed")
        self.assertSequenceEqual(edge_children, [])

    def test_create_updated_urdf_file(self):
        """
        Check that calling the constructor creates
        the updated urdf file.
        """

        # Delete the urdf file
        mini_cheetah_path_updated = self.mini_cheetah_path.parent / "miniCheetah_updated.urdf"
        os.remove(str(mini_cheetah_path_updated))
        self.assertFalse(os.path.exists(mini_cheetah_path_updated))

        # Rebuild it
        RobotGraph(self.mini_cheetah_path, 'package://yobotics_description/', 
                'mini-cheetah-gazebo-urdf/yobo_model/yobotics_description')
        self.assertTrue(os.path.exists(mini_cheetah_path_updated))

    def test_get_node_name_to_index_dict(self):
        """
        Check if all the indexes of the nodes in the dictionary
        are unique.
        """

        key = list(self.mini_cheetah_URDF.get_node_name_to_index_dict())
        get_nodes_index = []

        for key in key:
            index = self.mini_cheetah_URDF.get_node_name_to_index_dict()[key]
            get_nodes_index.append(index)

        self.assertTrue(pd.Index(get_nodes_index).is_unique)

    def test_get_node_index_to_name_dict(self):
        """
        Check the index_to_name dict by running making sure the
        index_to_name dict and the name_to_index dict are consistent.
        """

        index_to_name = list(self.mini_cheetah_URDF.get_node_index_to_name_dict())
        name_to_index = list(self.mini_cheetah_URDF.get_node_name_to_index_dict())
        get_nodes_index = []

        for key in name_to_index:
            index = self.mini_cheetah_URDF.get_node_name_to_index_dict()[key]
            get_nodes_index.append(index)

        self.assertEqual(index_to_name, get_nodes_index)

    def test_get_edge_index_matrix(self):
        """
        Check the dimensionality of the edge matrix.
        """

        edge_matrix = self.mini_cheetah_URDF.get_edge_index_matrix()

        self.assertEqual(edge_matrix.shape[0], 2)
        self.assertEqual(edge_matrix.shape[1], 32)

    def test_get_num_nodes(self):
        """
        Check that the number of nodes are correct. 
        """

        self.assertEqual(self.mini_cheetah_URDF.get_num_nodes(), 17)

    def test_get_edge_connections_to_name_dict(self):
        """
        Check the connections_to_name dict by running making sure the
        connections_to_name dict and the name_to_connections dict are 
        consistent.
        """

        connections_to_name = list(
            self.mini_cheetah_URDF.get_edge_connections_to_name_dict())
        name_to_connections = list(
            self.mini_cheetah_URDF.get_edge_name_to_connections_dict())

        result = []
        for key in name_to_connections:
            connections = self.mini_cheetah_URDF.get_edge_name_to_connections_dict(
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
            self.mini_cheetah_URDF.get_edge_name_to_connections_dict())
        all_connections = []

        # Get all connections from dictionary
        for key in name_to_connections:
            connections = self.mini_cheetah_URDF.get_edge_name_to_connections_dict(
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


class TestHeterogeneousRobotGraph(unittest.TestCase):

    def setUp(self):
        self.path_to_go1_urdf = Path(
            Path('.').parent, 'urdf_files', 'Go1', 'go1.urdf').absolute()

        self.GO1_HETERO_GRAPH = HeterogeneousRobotGraph(
            self.path_to_go1_urdf, 'package://go1_description/',
            'unitree_ros/robots/go1_description')

    def test_get_node_name_to_index_dict(self):
        """
        Test that the dictionary properly assigns indices to the nodes.
        """

        dict_actual = self.GO1_HETERO_GRAPH.get_node_name_to_index_dict()
        dict_desired = {
            'floating_base': 0,
            'FR_hip_joint': 0,
            'FR_thigh_joint': 1,
            'FR_calf_joint': 2,
            'FL_hip_joint': 3,
            'FL_thigh_joint': 4,
            'FL_calf_joint': 5,
            'RR_hip_joint': 6,
            'RR_thigh_joint': 7,
            'RR_calf_joint': 8,
            'RL_hip_joint': 9,
            'RL_thigh_joint': 10,
            'RL_calf_joint': 11,
            'FR_foot_fixed': 0,
            'FL_foot_fixed': 1,
            'RR_foot_fixed': 2,
            'RL_foot_fixed': 3
        }
        self.assertDictEqual(dict_actual, dict_desired)

    def test_get_node_index_to_name_dict(self):
        """
        Test that the dictionary properly tracks indexes to node
        names for the given type.
        """

        dict_actual = self.GO1_HETERO_GRAPH.get_node_index_to_name_dict('base')
        dict_desired = {
            0: 'floating_base' 
        }
        self.assertDictEqual(dict_actual, dict_desired)

        dict_actual = self.GO1_HETERO_GRAPH.get_node_index_to_name_dict('foot')
        dict_desired = {
            0: 'FR_foot_fixed',
            1: 'FL_foot_fixed',
            2: 'RR_foot_fixed',
            3: 'RL_foot_fixed'
        }
        self.assertDictEqual(dict_actual, dict_desired)

        dict_actual = self.GO1_HETERO_GRAPH.get_node_index_to_name_dict('joint')
        dict_desired = {
            0: 'FR_hip_joint',
            1: 'FR_thigh_joint',
            2: 'FR_calf_joint',
            3: 'FL_hip_joint',
            4: 'FL_thigh_joint',
            5: 'FL_calf_joint',
            6: 'RR_hip_joint',
            7: 'RR_thigh_joint',
            8: 'RR_calf_joint',
            9: 'RL_hip_joint',
            10: 'RL_thigh_joint',
            11: 'RL_calf_joint'
        }
        self.assertDictEqual(dict_actual, dict_desired)

    def test_get_num_of_each_node_type(self):
        """
        Test that we can properly count the number of each 
        type of node.
        """

        number_actual = self.GO1_HETERO_GRAPH.get_num_of_each_node_type()
        number_desired = [1, 12, 4]
        self.assertSequenceEqual(number_actual, number_desired)

    def test_get_edge_index_matrices(self):
        """
        Test that we construct the correct matrices for a
        heterogeneous graph.
        """

        bj, jb, jj, fj, jf = self.GO1_HETERO_GRAPH.get_edge_index_matrices()
        bj_des = np.array([[0, 0, 0, 0], [0, 3, 6, 9]])
        jb_des = np.array([[0, 3, 6, 9], [0, 0, 0, 0]])
        jj_des = np.array([[0, 1, 1, 2, 3, 4, 4, 5, 6, 7, 7, 8, 9, 10, 10, 11],
                           [1, 0, 2, 1, 4, 3, 5, 4, 7, 6, 8, 7, 10, 9, 11, 10]])
        fj_des = np.array([[0, 1, 2, 3], [2, 5, 8, 11]])
        jf_des = np.array([[2, 5, 8, 11], [0, 1, 2, 3]])
        numpy.testing.assert_array_equal(bj, bj_des)
        numpy.testing.assert_array_equal(jb, jb_des)
        numpy.testing.assert_array_equal(jj, jj_des)
        numpy.testing.assert_array_equal(fj, fj_des)
        numpy.testing.assert_array_equal(jf, jf_des)

    def test_get_edge_attr_matrices(self):
        """
        Test that we add the correct attributes to the edges.

        TODO: Eventually add more extensive test case for jj.
        """

        bj, jb, jj, fj, jf = self.GO1_HETERO_GRAPH.get_edge_attr_matrices()
        bj_des = np.array([[5.204, 0.0168128557, -0.0002296769, -0.0002945293, 0.063009565, -4.18731e-05, 0.0716547275],
                           [5.204, 0.0168128557, -0.0002296769, -0.0002945293, 0.063009565, -4.18731e-05, 0.0716547275],
                           [5.204, 0.0168128557, -0.0002296769, -0.0002945293, 0.063009565, -4.18731e-05, 0.0716547275],
                           [5.204, 0.0168128557, -0.0002296769, -0.0002945293, 0.063009565, -4.18731e-05, 0.0716547275]])
        jb_des = np.array([[5.204, 0.0168128557, -0.0002296769, -0.0002945293, 0.063009565, -4.18731e-05, 0.0716547275],
                           [5.204, 0.0168128557, -0.0002296769, -0.0002945293, 0.063009565, -4.18731e-05, 0.0716547275],
                           [5.204, 0.0168128557, -0.0002296769, -0.0002945293, 0.063009565, -4.18731e-05, 0.0716547275],
                           [5.204, 0.0168128557, -0.0002296769, -0.0002945293, 0.063009565, -4.18731e-05, 0.0716547275]])
        jj_des_0_3 = np.array([[0.591, 0.000334008405, 1.0826066e-05, 1.290732e-06, 0.000619101213, -1.643194e-06, 0.00040057614],
                           [0.591, 0.000334008405, 1.0826066e-05, 1.290732e-06, 0.000619101213, -1.643194e-06, 0.00040057614],
                           [0.92, 0.004431760472, -5.7496807e-05, -0.000218457134, 0.004485671726, -0.000572001265, 0.000740309489],
                           [0.92, 0.004431760472, -5.7496807e-05, -0.000218457134, 0.004485671726, -0.000572001265, 0.000740309489]])
        fj_des = np.array([[0.135862, 0.001088793059, -2.55679e-07, 7.117814e-06, 0.001100428748, 2.077264e-06, 2.4787446e-05],
                           [0.135862, 0.001088793059, -2.55679e-07, 7.117814e-06, 0.001100428748, 2.077264e-06, 2.4787446e-05],
                           [0.135862, 0.001088793059, -2.55679e-07, 7.117814e-06, 0.001100428748, 2.077264e-06, 2.4787446e-05],
                           [0.135862, 0.001088793059, -2.55679e-07, 7.117814e-06, 0.001100428748, 2.077264e-06, 2.4787446e-05]])
        jf_des = np.array([[0.135862, 0.001088793059, -2.55679e-07, 7.117814e-06, 0.001100428748, 2.077264e-06, 2.4787446e-05],
                           [0.135862, 0.001088793059, -2.55679e-07, 7.117814e-06, 0.001100428748, 2.077264e-06, 2.4787446e-05],
                           [0.135862, 0.001088793059, -2.55679e-07, 7.117814e-06, 0.001100428748, 2.077264e-06, 2.4787446e-05],
                           [0.135862, 0.001088793059, -2.55679e-07, 7.117814e-06, 0.001100428748, 2.077264e-06, 2.4787446e-05]])
        numpy.testing.assert_array_equal(bj, bj_des, verbose=True)
        numpy.testing.assert_array_equal(jb, jb_des, verbose=True)
        numpy.testing.assert_array_equal(jj[0:4], jj_des_0_3, verbose=True)
        numpy.testing.assert_array_equal(fj, fj_des, verbose=True)
        numpy.testing.assert_array_equal(jf, jf_des, verbose=True)

if __name__ == '__main__':
    unittest.main()
