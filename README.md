# state-estimation-gnn
This repository implements a graph neural network for estimating the GRF values on the feet of a quadruped robot.
Ultimately, this can be used in robotic state estimation and control.

## Environment Setup
To get started, setup a Conda Python environment with Python=3.11, and run the following command to install the necessary dependencies:
```
pip install -r requirements.txt
```

## Installation
Then, install the library with the following command:
```
pip install .
```

## URDF Download
The necessary URDF files are part of git submodules in this repository, so run the following commands to download them:
```
git submodule init
git submodule update
```

## Training a new model
To train a new model from QuadSDK data, run the following command within your Conda environment:

```
python research/train.py
```

First, this command will process the dataset to ensure quick data access during training. Next, it will begin a 
training a Heterogeneous GNN and log the results to WandB (Weights and Biases). Note that you may need to put in
a WandB key in order to use this logging feature. You can disable this logging following the instructions in 
[#Editing this repository](#editing-this-repository), since the logger code is found on line 386 of 
`src/grfgnn/gnnLightning.py` and can be commented out.

The model weights with the best validation MSE loss will be saved in the following folder, based on the model 
type and the randomly chosen model name (which is output in the terminal when training begins):
```
<repository base directory>/models/<model-type>-<model_name>/
```

## Evaluating a trained model

If you used logging on a previous step, you can see the losses and other relevant info in WandB (Weights and Biases).

But, regardless, whether you used logging or not, you can evaluate the data on the test subset of the Quad-SDK data 
and see the predicted and ground truth GRF values for all four legs.

First, edit the file `research/evaluator.py` following the provided comments; this will tell the code what model you want to visualize, and how many entries in the dataset to use.

Then, run the following command to evaluate on the model:
```
python research/evaluator.py
```

The visualization of the predicted and GT GRF will be found in a file called `model_eval_results.pdf`.

## Changing the model type
Currently, three model types are supported:
- `mlp`
- `gnn`
- `heterogeneous_gnn`

To change the model type, please change the `model_type` parameter in the `train.py` and `evaluator.py` files.

## Editing this repository

If you want to make changes to the source files, feel free to edit them in the `src/grfgnn` folder, and then 
rebuild the library following the instructions in [#Installation](#installation).