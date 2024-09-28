# MI-HGNN for contact estimation/classification
This repository implements a Morphology-Inspired Heterogeneous Graph Neural Network (MI-HGNN) for estimating contact information on the feet of a quadruped robot. For more details, see our publication "[MI-HGNN: Morphology-Informed Heterogeneous Graph Neural Network for Legged Robot Contact Perception](https://arxiv.org/abs/2409.11146)".

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

## Replicating Paper Experiments

To replicate the experiments referenced in our paper or access our trained model weights, see `paper/README.md`.

## Editing and Contributing

Although in our paper, we only applied the MI-HGNN on quadruped robots for contact perception, it can also be applied to other multi-body dynamical systems and on other tasks/datasets. New URDF files can be added by following the instructions in `urdf_files/README.md`. Datasets can be found in the `src/mi_hgnn/datasets_py` directory, and model definitions and training code can be found in the `src/mi_hgnn/lightning_py` directory. We encourage you to extend the library for your own applications. Please reference [#Replicating-Paper-Experiments](#replicating-paper-experiments) for examples on how to train and evaluate models with our repository.

After making changes, rebuild the library following the instructions in [#Installation](#installation). To make sure that your changes haven't
broken critical functionality, run the test cases found in the `tests` directory.

If you'd like to contribute to the repository, write sufficient and necessary test cases for your additions in the `tests` directory, and then open a pull request.

## Citation

If you find our repository or our work useful, please cite the relevant publication:

```
@article{butterfield2024mi,
  title={MI-HGNN: Morphology-Informed Heterogeneous Graph Neural Network for Legged Robot Contact Perception},
  author={Butterfield, Daniel and Garimella, Sandilya Sai and Cheng, Nai-Jen and Gan, Lu},
  journal={arXiv preprint arXiv:2409.11146},
  year={2024},
  eprint={2409.11146},
  url={https://arxiv.org/abs/2409.11146},
}
```

## Contact / Issues

For any issues with this repository, feel free to open an issue on GitHub. For other inquiries, please contact Daniel Butterfield (dbutterfield3@gatech.edu) or the Lunar Lab (https://sites.gatech.edu/lunarlab/).