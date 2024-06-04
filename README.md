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

## Dataset Installation
Next, install Dr. Xiong's Quadruped dataset by going to the following link (https://gtvault-my.sharepoint.com/:u:/g/personal/lgan31_gatech_edu/Ee5lmlVVQTZCreMujfQOTFABPJn6RyjK8UDABFXPL86UcA?e=tBGhhO), unzipping the folder, and then placing all of the bag
files within the following folder:
```
<repository base directory>/datasets/xiong_simulated/raw/
```

So, for example, this should be a valid path to a bag file:
```
<repository base directory>/datasets/xiong_simulated/raw/traj_0000.bag
```
There should be about 100 bag files.

## Training a new model
To train a new model from Dr. Xiong's quadruped data, run the following command within your Conda environment:

```
python research/train.py
```

First, this command will process the dataset to ensure quick data access during training. Next, it will begin a 
training a GNN and log the results to WandB (Weights and Biases). Note that you may need to put in
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

But, regardless, whether you used logging or not, you can evaluate the data on the test subset of Dr. Xiong's quadruped data 
and see the predicted and ground truth GRF values for all four legs.

First, edit the file `research/evaluator.py` on lines 22, 23, and 46; this will tell the code what model you want to visualize, and how many entries in the dataset to use.

Then, run the following command to evaluate on the model:
```
python research/evaluator.py
```

The visualization of the predicted and GT GRF will be found in a file called `model_eval_results.pdf`.

## Editing this repository

If you want to make changes to the model type, the training parameters, or anything else, modify the files
found in the `src/grfgnn` folder, and then rebuild the library following the instructions in [#Installation](#installation).

Currently, two model types are supported:
- `mlp`
- `gnn`
To change the model type, please change line 316 in `src/grfgnn/gnnLightning.py`.
