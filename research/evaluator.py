from grfgnn.datasets import visualize_graph, CerberusTrackDataset, CerberusStreetDataset, CerberusCampusDataset, Go1SimulatedDataset
from grfgnn.graphParser import NormalRobotGraph
from pathlib import Path
from grfgnn.gnnLightning import evaluate_model_and_visualize
import torch


path_to_go1_urdf = Path(Path('.').parent, 'urdf_files', 'Go1',
                        'go1.urdf').absolute()
path_to_xiong_simulated = Path(
        Path('.').parent, 'datasets', 'xiong_simulated').absolute()

# Load URDF files
GO1_URDF = NormalRobotGraph(path_to_go1_urdf, 'package://go1_description/',
                     'unitree_ros/robots/go1_description', True)

# Define model type
model_type = 'gnn'

# Give path to checkpoint
# ================================= FILL THESE IN ===================================
model_directory_name = None # Ex: 'gnn-Jared-Wright'
ckpt_file_name = None       # Ex: 'epoch=0-val_loss=6544.70.ckpt'
# ===================================================================================
path_to_checkpoint = None
if model_type == 'gnn':
    path_to_checkpoint = Path(
        Path('.').parent, 'models', model_directory_name, ckpt_file_name)

# Initalize the dataset
go1_sim_dataset = Go1SimulatedDataset(
    path_to_xiong_simulated, path_to_go1_urdf, 'package://go1_description/',
    'unitree_ros/robots/go1_description', model_type)

# Evaluate the model
# ================================= FILL THESE IN ===================================
num_entries_to_visualize = None # Ex: 2000
# ===================================================================================
evaluate_model_and_visualize(model_type, path_to_checkpoint, go1_sim_dataset,
                             (0, num_entries_to_visualize), "model_eval_results.pdf")
