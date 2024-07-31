# Paper Replication

This directory provides the model weights for all of our MI-HGNN models referenced in the paper. Whenever a specific trained model is referenced in this README (for example, `ancient-salad-5`), it will be highlighted as shown, and there will be a folder on [Google Drive](https://drive.google.com/drive/folders/1NS5H_JIXW-pORyQUR15-t3mzf2byG26v?usp=sharing) with its name. In that folder will be its weights after the full 30 epochs of training, which were used to generate the paper results.

To find the name of a specific model referenced in the paper or to replicate the results, refer to the following sections below which correspond to paper sections.

## Contact Detection (Classification) Experiment

To replicate the results of these experiments on your own end, input the checkpoint path into the `evaluator_classification.py` file found in the `research` directory of this repository.

As of the time of this commit, the main experiment has not been completed yet.

### Abalation Study

For this paper, we conducted an abalation study to see how parameter-efficient our model is. In the paper, we give the each trained model's layer number, hidden size, parameter number, and finally, the state accuracy on the test set. The table below relates these parameters to specific trained model names so that you can find the exact checkpoint weights for each model.

| Number of Layers | Hidden Sizes | Number of Parameters | State Accuracy (Test) | Model Name            |
| ---------------- | ------------ | ---------------------| --------------------- | --------------------- |
| 4                | 5            | **11,621**           | 85.76                 | `restful-cosmos-25`   |
| 4                | 10           | 25,241               | 85.71                 | `distinctive-river-24`|
| 4                | 25           | 78,101               | 86.92                 | `noble-bird-23`       |
| 4                | 50           | 206,201              | 86.64                 | `toasty-resonance-22` |
| 4                | 128          | 927,233              | 87.12                 | `ancient-dust-37`     |
| 8                | 50           | 307,201              | 88.81                 | `pious-donkey-28`     |
| 8                | 128          | 1,585,153            | **90.5**              | `sunny-pond-26`       |
| 8                | 256          | 5,791,745            | 90.13                 | `dauntless-brook-29`  |
| 12               | 50           | 408,201              | 90.04                 | `twilight-sponge-32`  |
| 12               | 128          | 2,243,073            | 90.21                 | `super-grass-6`       |
| 12               | 256          | 8,418,305            | 14.15                 | `wandering-moon-31`   |

### Side note on normalization

In our paper, we mention that we found that our MI-HGNN model performed better without entry-wise normalization. We found this by running the two models below. This wasn't an exhausive experiment, which is why it only deserved a short reference in the paper. However, you can see that for this specific configuration of layer size and hidden size, our MI-HGNN model has a 2.71% increase in accuracy when disabling the entry-wise normalization used in this experiment.

| Number of Layers | Hidden Sizes | Normalization | State Accuracy (Test) | Model Name        |
| ---------------- | ------------ | --------------| --------------------- | ----------------- |
| 12               | 128          | False         | 92.92                | `ancient-salad-5` |
| 12               | 128          | True          | 90.21                | `super-grass-6`   |

## Ground Reaction Force Estimation (Regression) Experiment 

As of this time of this commit, this experiment has not been completed yet.