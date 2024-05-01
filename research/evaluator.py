from grfgnn.datasets import visualize_graph, CerberusTrackDataset, CerberusStreetDataset
from grfgnn.urdfParser import RobotURDF
from pathlib import Path
from grfgnn.gnnLightning import visualize_model_outputs

path_to_urdf = Path(Path('.').parent, 'urdf_files', 'A1', 'a1.urdf').absolute()
path_to_cerberus_street = Path(Path('.').parent, 'datasets', 'cerberus_street').absolute()
path_to_cerberus_track = Path(Path('.').parent, 'datasets', 'cerberus_track').absolute()
#path_to_checkpoint = Path(Path('.').parent, 'models', TODO, 'epoch=6-val_loss=5764.05.ckpt')

# Load the A1 urdf
A1_URDF = RobotURDF(path_to_urdf, 'package://a1_description/',
                    'unitree_ros/robots/a1_description', True)
model_type = 'gnn'

# Initalize the datasets
street_dataset = CerberusStreetDataset(
    path_to_cerberus_street,
    A1_URDF, model_type)
track_dataset = CerberusTrackDataset(
    path_to_cerberus_track,
    A1_URDF, model_type)

visualize_graph(track_dataset[0], A1_URDF, "temp.pdf")

#visualize_model_outputs(model_type, path_to_checkpoint, path_to_urdf, 
 #                       path_to_cerberus_track, 10000,
 #                       Path('.', 'research', 'pred_new_' + model_type + '.png'))