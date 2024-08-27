# Paper Replication

This directory provides the model weights for all of our MI-HGNN models (and MLP models) referenced in the paper. Whenever a specific trained model is referenced in this README (for example, `ancient-salad-5`), it will be highlighted as shown, and there will be a folder on [Google Drive](https://drive.google.com/drive/folders/1NS5H_JIXW-pORyQUR15-t3mzf2byG26v?usp=sharing) with its name. In that folder will be its weights after the full 30 epochs of training, which were used to generate the paper results.

To find the name of a specific model referenced in the paper or to replicate the results, refer to the following sections below which correspond to paper sections.

## Contact Detection (Classification) Experiment

To replicate the results of these experiments on your own end, input the checkpoint path into the `evaluator_classification.py` file found in the `research` directory of this repository.

As of the time of this commit, the main experiment has not been completed yet.

### Abalation Study

For this paper, we conducted an abalation study to see how parameter-efficient our model is. In the paper, we give the each trained model's layer number, hidden size, parameter number, and finally, the state accuracy on the test set. The table below relates these parameters to specific trained model names so that you can find the exact checkpoint weights for each model.

| Number of Layers | Hidden Sizes | Number of Parameters | State Accuracy (Test) | Model Name            |
| ---------------- | ------------ | ---------------------| --------------------- | --------------------- |
| 4                | 5            |                      |                       |                       |
| 4                | 10           |                      |                       |                       |
| 4                | 25           |                      |                       |                       |
| 4                | 50           |                      |                       |                       |
| 4                | 128          |                      |                       |                       |
| 8                | 50           |                      |                       |                       |
| 8                | 128          |                      |                       |                       |
| 8                | 256          |                      |                       |                       |
| 12               | 50           |                      |                       |                       |
| 12               | 128          |                      |                       |                       |
| 12               | 256          |                      |                       |                       |

### Side note on normalization

In our paper, we mention that we found that our MI-HGNN model performed better without entry-wise normalization. We found this by running the two models below. This wasn't an exhausive experiment, which is why it only deserved a short reference in the paper. However, you can see that for this specific configuration of layer size and hidden size, our MI-HGNN model has a _______ increase in accuracy when disabling the entry-wise normalization used in this experiment.

| Number of Layers | Hidden Sizes | Normalization | State Accuracy (Test) | Model Name        |
| ---------------- | ------------ | --------------| --------------------- | ----------------- |
| 12               | 128          | False         |                       |                   |
| 12               | 128          | True          |                       |                   |

## Ground Reaction Force Estimation (Regression) Experiment 

As of this time of this commit, this experiment has not been completed yet.