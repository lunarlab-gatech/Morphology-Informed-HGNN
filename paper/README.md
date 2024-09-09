# Paper Replication

This directory provides the model weights for all of our MI-HGNN models referenced in the paper. Whenever a specific trained model is referenced in this README (for example, `ancient-salad-5`), it will be highlighted as shown, and there will be a folder on Georgia Tech's [Dropbox](https://www.dropbox.com/scl/fo/8p165xcfbdfwlcr3jx7tb/ABoxs5BOEXsQnJgFXF_Mjcc?rlkey=znrs7oyu29qsswpd3a5r55zk8&st=53v30ys3&dl=0) with its name. Unless otherwise specified, the model weights used for the paper were those trained the longest (have highest `epoch=` number in their .ckpt file).

To find the name of a specific model referenced in the paper or to replicate the results, refer to the following sections below which correspond to paper sections.

## Contact Detection (Classification) Experiment

Our models trained during this experiment can be found in the table below. For more details, see the `contact_experiment.csv` file in this directory. To evaluate the model metrics on your own end, input the checkpoint path into the `evaluator_classification.py` file found in the `research` directory of this repository:

| Number of Layers | Hidden Sizes | Seed | State Accuracy (Test) | Model Name            |
| ---------------- | ------------ | ---- |---------------------- | --------------------- |
| 8                | 128          |    0 | 0.874120593070984     | `gentle-morning-4`     |
| 8                | 128          |    1 | 0.895811080932617     | `leafy-totem-5`         |
| 8                | 128          |    2 | 0.868574500083923     | `different-oath-6`      |
| 8                | 128          |    3 | 0.878039181232452     | `hopeful-mountain-7`    |
| 8                | 128          |    4 | 0.855807065963745     | `revived-durian-8`      |
| 8                | 128          |    5 | 0.875732064247131     | `robust-planet-9`       |
| 8                | 128          |    6 | 0.883218884468079     | `super-microwave-10`    |
| 8                | 128          |    7 | 0.880922436714172     | `valiant-dawn-11`       |

The baseline models we compared to (ECNN, CNN-aug, CNN) were trained on this release: [MorphoSymm-Replication -> With Bug Fixes](https://github.com/lunarlab-gatech/MorphoSymm-Replication/releases/tag/RepBugFixes). See that repository for information on accessing those model weights, and replicating the Contact Detection Experiment Figure seen in our paper.

### Abalation Study

We conducted an abalation study to see how parameter-efficient our model is. In the paper, we give the each trained model's layer number, hidden size, parameter number, and finally, the state accuracy on the test set. Here those values are associated with the model's name in the table below. For more details, see the `contact_experiment_ablation.csv` file in this directory.

| Number of Layers | Hidden Sizes | Model Name             |
| ---------------- | ------------ | ---------------------- |
| 4                | 5            | `prime-water-16`       |
| 4                | 10           | `driven-shape-17`      |
| 4                | 25           | `autumn-terrain-18`    |
| 4                | 50           | `comfy-dawn-19`        |
| 4                | 128          | `prime-butterfly-20`   |
| 4                | 256          | `youthful-galaxy-21`   |
| 8                | 50           | `exalted-mountain-22`  |
| 8                | 128          | `serene-armadillo-23`  |
| 8                | 256          | `playful-durian-12`    |
| 12               | 50           | `twilight-armadillo-15`|
| 12               | 128          | `sparkling-music-14`   |
| 12               | 256          | `stoic-mountain-13`    |


## Ground Reaction Force Estimation (Regression) Experiment 

As of this time of this commit, this experiment has not been completed yet.