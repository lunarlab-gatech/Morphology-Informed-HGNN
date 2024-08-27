from urchin import URDF
from pathlib import Path
import numpy as np
import pinocchio as pin
from grfgnn import QuadSDKDataset_A1Speed0_5, QuadSDKDataset_A1Speed1_0, QuadSDKDataset_A1Speed1_5FlippedOver
from os.path import dirname, join, abspath
import matplotlib.pyplot as plt

robot = URDF.load('/Users/jackson/Documents/state-estimation-gnn/urdf_files/A1/a1.urdf')


