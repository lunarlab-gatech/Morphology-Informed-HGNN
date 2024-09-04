# MI-HGNN for contact estimation/classification
This repository implements a Morphology-Inspired Heterogeneous Graph Neural Network (MI-HGNN) for estimating contact information on the feet of a quadruped robot.

## Installation
To get started, setup a Conda Python environment with Python=3.11:
```
conda create -n mi-hgnn python=3.11
conda activate mi-hgnn
```

Then, install the library (and dependencies) with the following command:
```
pip install .
```

Note, if you have any issues with setup, refer to `environment_files/README.md` so you can install the exact libraries we used.

## URDF Download
The necessary URDF files are part of git submodules in this repository, so run the following commands to download them:
```
git submodule init
git submodule update
```

## Training models

Follow the commands below given to train your model below for your specific scenario. Note that you may need to put in
a WandB key in order to log results to Weights & Biases. You can disable this logging by enabling `disable_logger`
in the `train_model` function.

The model weights will be saved in the following folder, based on the model 
type and the randomly chosen model name (which is output in the terminal when training begins):
```
<repository base directory>/models/<model_name>/
```
There will be the six models saved, one with the final model weights, and five with the best validation losses during training.

### Contact Detection (Classification) Experiment

To train a model from the dataset from [MorphoSymm-Replication]([https://arxiv.org/abs/2106.15713](https://github.com/lunarlab-gatech/MorphoSymm-Replication/releases/tag/RepBugFixes)), run the following command within your Conda environment. Feel free to edit the model parameters within the file itself:

```
python research/train_classification.py
```

To evaluate a model, edit `evaluator_classification.py` to specify which model to evaluate. Then, run the following command:

```
python research/evaluator_classification.py
```

### GRF (Regression) Model

Not Yet Implemented.

### Your Own Custom Model

Tutorial not yet written.

## Editing this repository
If you want to make changes to the source files, feel free to edit them in the `src/mi_hgnn/' folder, and then 
rebuild the library following the instructions in [#Installation](#installation).

## Paper Replication
To replicate our paper results with the model weights we trained, see `paper/README.md`.
