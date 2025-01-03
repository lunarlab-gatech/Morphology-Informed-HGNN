# URDF files

Each robot URDF file with its corresponding repository can be found in a folder with its name. As robots can have multiple URDF files (for example, with different naming conventions, frame definitions, etc), double check any URDF in this directory before you use it to make sure it has the data you expect. For example, "A1" refers to the URDF file from [unitreerobotics](https://github.com/unitreerobotics/unitree_ros/tree/master/robots/a1_description), whereas "A1-Quad" refers to the URDF file from our fork of [quad-sdk](https://github.com/lunarlab-gatech/quad_sdk_fork/tree/a1). 

Note, the ```urdfParser.py``` file will take the ```*.urdf``` file and generate a ```*_updated.urdf``` file in the same directory, which contains updated paths based on your current system. For this reason, these files are not commited to the GitHub repo, as they should vary per device.

## Adding a new URDF file

Before you add a new URDF file, you need to make sure that it has all of it's non-kinematic nodes pruned. If it doesn't, you'll have to generate a new URDF file without them. See [Generating new URDF files](#generating-new-urdf-files) for more information.

Once you have a new URDF file, make sure to create a new folder for it and add the URDF. Add the repository that the URDF depends on to the folder as a git submodule. Finally, if you desire, you can add the PDF for the URDF following the instructions in [Generating PDF](#generating-pdf).

### Generating new URDF files

First, make sure to install ROS and the corresponding repository that comes along with the URDF file (which I'll call the URDF repository). Build the repository using ROS, and make sure it builds without errors. You may need to install more dependencies, depending on what the URDF repository says.

Next, find the xacro file used to create the URDF. This will most likely be found in the URDF repository. Manually comment out any non-kinematic structures. Why is this necessary? Our MI-HGNN relies upon the input graph being morphology-informed, which we define as graph nodes representing kinematic joints and graph edges representing kinematic links. Thus, it assumes that the input graph is composed of all of the robots kinematic joints, and that no other fixed structures are included (like cameras or IMUs). Failing to remove any non-kinematic structures will cause unexpected effects, likely decreasing model performance. An easy way to make sure you have done this properly is by [generating a PDF](#generating-pdf) of the URDF file after you create it.

Run the following command to generate the new URDF file:

```
rosrun xacro xacro <xacro_path> > <new_urdf_path>
```


### Generating PDF
To generate the pdf file that corresponds to the urdf file, install ROS 1, and run the following command:

```
sudo apt-get install graphviz
```

Next, follow the instructions here: https://manpages.ubuntu.com/manpages/jammy/man1/urdf_to_graphiz.1.html

