from pathlib import Path
from grfgnn.gnnLightning import train_model, train_model_go1_simulated

path_to_urdf = Path('urdf_files', 'Go1', 'go1.urdf').absolute()
path_to_xiong_simulated = Path(
        Path('.').parent, 'datasets', 'xiong_simulated').absolute()

train_model_go1_simulated(path_to_urdf, path_to_xiong_simulated)