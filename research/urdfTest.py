from grfgnn import NormalRobotGraph, HeterogeneousRobotGraph
from pathlib import Path

path_to_a1_urdf = Path(Path('.').parent, 'urdf_files', 'A1',
                       'a1.urdf').absolute()
# A1_URDF = NormalRobotGraph(path_to_a1_urdf, 'package://a1_description/',
#                           'unitree_ros/robots/a1_description', True)

path_to_go1_urdf = Path(Path('.').parent, 'urdf_files', 'Go1',
                        'go1.urdf').absolute()
# GO1_URDF = NormalRobotGraph(path_to_go1_urdf, 'package://go1_description/',
#                            'unitree_ros/robots/go1_description', True)

A1_HETERO_GRAPH = HeterogeneousRobotGraph(
    path_to_a1_urdf, 'package://a1_description/',
    'unitree_ros/robots/a1_description', True)
A1_HETERO_GRAPH.display_graph_info()
print(A1_HETERO_GRAPH.get_node_name_to_index_dict())
