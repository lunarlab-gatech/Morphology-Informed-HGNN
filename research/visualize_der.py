from grfgnn import visualize_derivatives, QuadSDKDataset
from pathlib import Path

path_to_urdf = Path('urdf_files', 'A1', 'a1.urdf').absolute()
path_to_quad_sdk = Path(Path('.').parent, 'datasets', 'QuadSDK-A1Speed1.0').absolute()
quad_sdk_dataset = QuadSDKDataset(path_to_quad_sdk, path_to_urdf, 'package://a1_description/', 'unitree_ros/robots/a1_description', 'heterogeneous_gnn')
visualize_derivatives(quad_sdk_dataset)