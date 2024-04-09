from pathlib import Path
from grfgnn.gnnLightning import train_model

path_to_urdf = Path('urdf_files', 'A1', 'a1.urdf').absolute()
path_to_cerberus_street = Path('datasets', 'cerberus_street').absolute()
path_to_cerberus_track = Path('datasets', 'cerberus_track').absolute()

train_model(path_to_urdf, path_to_cerberus_street, path_to_cerberus_track, 'mlp')