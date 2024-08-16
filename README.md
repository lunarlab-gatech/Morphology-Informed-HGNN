# MI-HGNN for contact estimation/classification
This repository implements a Morphologically-inspired Heterogeneous graph neural network for estimating contact information on the feet of a quadruped robot.

## Installation
To get started, setup a Conda Python environment with Python=3.11:
```
conda create -n mi-hgnn python=3.11
```

Then, install the library (and dependencies) with the following command:
```
pip install .
```

Note, if you have any issues with setup, refer to the `environment_files/README.md` so you can install the exact libraries we used.

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
<repository base directory>/models/<model-type>-<model_name>/
```
There will be the six models saved, one with the final model weights, and five with the best validation losses during training.

### LinTzuYaun Contact Dataset Model

To train a model from the dataset from [Legged Robot State Estimation using Invariant Kalman Filtering and Learned Contact Events](https://arxiv.org/abs/2106.15713), run the following command within your Conda environment:

```
python research/train_classification.py
```

If you want to customize the model used, the number of layers, or the hidden size, feel free to change the corresponding variables.

To evaluate this model, edit `evaluator_classification.py` to specify which model to evaluate, its type, and the number of dataset entries to consider. Then, run the following command:

```
python research/evaluator_classification.py
```

The visualization of the predicted and GT values will be found in a file called `model_eval_results.pdf` in the same directory as your model weights.

### QuadSDK & Real World GRF Model

Not Yet Implemented.

### Your Own Custom Model

Tutorial not yet written.


## Changing the model type
Currently, two model types are supported:
- `mlp`
- `heterogeneous_gnn`
To change the model type, please change the `model_type` parameter in the `train.py` and `evaluator.py` files.

## Editing this repository
If you want to make changes to the source files, feel free to edit them in the `src/grfgnn` folder, and then 
rebuild the library following the instructions in [#Installation](#installation).

## Paper Replication
To replicate our paper results with the model weights we trained, see `paper/README.md`.