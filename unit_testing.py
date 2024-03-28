import unittest
import os
import urdfParser
import pandas as pd
import numpy as np

class TestStringMethods(unittest.TestCase):

    def test___init__(self):
        """
        check if self.nodes has all the name in the URDF file (for False)
        check if self.edges has all the name in the URDF file (for True)

        """
        nodes_name = {'world','base_link','trunk',
                      'lf_hipassembly','lh_hipassembly','rf_hipassembly','rh_hipassembly',
                      'lf_upperleg','lh_upperleg','rf_upperleg','rh_upperleg',
                      'lf_lowerleg','lh_lowerleg','rf_lowerleg','rh_lowerleg',
                      'lf_foot','lh_foot','rf_foot','rh_foot'}
        
        edges_name = {'floating_base_joint','floating_base',
                      'lf_haa_joint','lh_haa_joint','rf_haa_joint','rh_haa_joint',
                      'lf_hfe_joint','lh_hfe_joint','rf_hfe_joint','rh_hfe_joint',
                      'lf_kfe_joint','lh_kfe_joint','rf_kfe_joint','rh_kfe_joint',
                      'lf_foot_joint','lh_foot_joint','rf_foot_joint','rh_foot_joint'}
        
        
        HyQ_URDF = urdfParser.RobotURDF('urdf_files\HyQ\hyq.urdf',
                                        'package://hyq_description/', 'hyq-description',False)
        
        get_nodes_name = HyQ_URDF.nodes

        for get_nodes_name in (get_nodes_name):
            if get_nodes_name.name not in nodes_name:
                print(get_nodes_name.name)

        HyQ_URDF = urdfParser.RobotURDF('urdf_files\HyQ\hyq.urdf',
                                        'package://hyq_description/', 'hyq-description',True)
        
        get_edges_name = HyQ_URDF.nodes

        for get_edges_name in (get_edges_name):
            if get_edges_name.name not in edges_name:
                print('False')
        
        # raise NotImplemented

    def test_create_updated_urdf_file(self):
        """
        check if the updated urdf file exist

        """
        actual_url = os.path.join(os.getcwd(), os.path.dirname('urdf_files\HyQ\hyq.urdf'),
                                  'package://hyq_description/', "temp")[:-4]
        os.remove("actual_url")
        HyQ_URDF = urdfParser.RobotURDF('urdf_files\HyQ\hyq.urdf',
                                        'package://hyq_description/', 'hyq-description', False)

        self.assertTrue(os.path.exists(actual_url))
        
        # raise NotImplemented

    def test_get_node_name_to_index_dict(self):
        """
        check if the index is unique

        """
        HyQ_URDF = urdfParser.RobotURDF('urdf_files\HyQ\hyq.urdf',
                                        'package://hyq_description/', 'hyq-description',False)
        
        key = list(HyQ_URDF.get_node_name_to_index_dict())
        get_nodes_index = []

        for key in key:
            index = HyQ_URDF.get_node_name_to_index_dict()[key]
            get_nodes_index.append(index)

        result = pd.Index(get_nodes_index).is_unique
        # print(result)

        if result == False:
            raise NotImplemented
        else:
            return result
    
    def test_get_node_index_to_name_dict(self):
        """
        check if the index matchse the dictionary
        or the name is unique
        (maybe call test_get_node_name_to_index_dict would be easier to check)

        """
        HyQ_URDF = urdfParser.RobotURDF('urdf_files\HyQ\hyq.urdf',
                                        'package://hyq_description/', 'hyq-description',False)
        
        key_index = list(HyQ_URDF.get_node_index_to_name_dict())
        
        # get_nodes_index = self.test_get_node_name_to_index_dict()

        key = list(HyQ_URDF.get_node_name_to_index_dict())
        get_nodes_index = []

        for key in key:
            index = HyQ_URDF.get_node_name_to_index_dict()[key]
            get_nodes_index.append(index)

        self.assertEqual(key_index, get_nodes_index)

        # raise NotImplemented
    
    def test_get_edge_index_matrix(self):
        """
        check the dimension of the edge matrix 

        """
        HyQ_URDF = urdfParser.RobotURDF('urdf_files\HyQ\hyq.urdf',
                                        'package://hyq_description/', 'hyq-description')
        edge_matrix = HyQ_URDF.get_edge_index_matrix()
        # print(edge_matrix.shape)

        m = edge_matrix.shape[0]
        n = edge_matrix.shape[1]
        num_of_edges = HyQ_URDF.get_num_nodes() - 1

        self.assertEqual(m, 2)
        self.assertEqual(2*num_of_edges, n)

        # raise NotImplemented
    
    def test_get_num_nodes(self):
        """
        check the number of the node(False)/ edges(True)

        """
        HyQ_URDF = urdfParser.RobotURDF('urdf_files\HyQ\hyq.urdf',
                                        'package://hyq_description/', 'hyq-description', False)
        a = HyQ_URDF.get_num_nodes()
        b = 19
        self.assertEqual(a, b)
        
        HyQ_URDF = urdfParser.RobotURDF('urdf_files\HyQ\hyq.urdf',
                                        'package://hyq_description/', 'hyq-description', True)
        a = HyQ_URDF.get_num_nodes()
        b = 18
        self.assertEqual(a, b)

        # raise NotImplemented
    
    def test_get_edge_connections_to_name_dict(self):
        """
        check if the index matchse the dictionary 
        or the name is unique
        (maybe call test_get_edge_name_to_connections_dict would be easier to check)

        """
        HyQ_URDF = urdfParser.RobotURDF('urdf_files\HyQ\hyq.urdf',
                                        'package://hyq_description/', 'hyq-description',False)
        
        expected_index = list(HyQ_URDF.get_edge_connections_to_name_dict())
        print(expected_index)

        key = list(HyQ_URDF.get_edge_name_to_connections_dict())
        
        connections_dict = []

        for key in key:
            index = HyQ_URDF.get_edge_name_to_connections_dict()[key]
            for i in range (index.shape[1]):
                real_index = np.squeeze(index[:,i].reshape(1,-1))
                connections_dict.append(real_index)

        connections_dict = [tuple(arr) for arr in connections_dict]
        print(connections_dict)

        self.assertEqual(expected_index, connections_dict)

        # raise NotImplemented

    def test_get_edge_name_to_connections_dict(self):
        """
        check the dictionary is unique

        """
        HyQ_URDF = urdfParser.RobotURDF('urdf_files\HyQ\hyq.urdf',
                                        'package://hyq_description/', 'hyq-description',False)
        
        key = list(HyQ_URDF.get_edge_name_to_connections_dict())
        connections_dict = []

        for key in key:
            index = HyQ_URDF.get_edge_name_to_connections_dict()[key]
            connections_dict.append(index)
        # print(connections_dict)
            
        seen_arrays = set()
        for array in connections_dict:
        # Convert the array to a tuple since lists are not hashable
            array_tuple = tuple(array)

        # Check if the array is already seen
        if array_tuple in seen_arrays:
            return False

if __name__ == '__main__':
    unittest.main()