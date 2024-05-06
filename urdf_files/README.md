# URDF files

Each robot URDF file with its corresponding repository can be found in a folder with its name.

Note, the ```urdfParser.py``` file will take the ```*.urdf``` file and generate a ```*_updated.urdf``` file in the same directory, which contains updated paths based on your current system. For this reason, these files are not commited to the GitHub repo, as they should vary per device.

## Adding a new URDF file

Before you add a new URDF file, you need to make sure that it has all of it's non-kinematic nodes pruned. If it doesn't you'll have to generate a new URDF file without them. See [Generating new URDF files](#generating-new-urdf-files) for more information.

Once you have a new URDF file, make sure to create a new folder for it and add the URDF. Add the repository that the URDF depends on to the folder as a git submodule. Finally, if you want to, add the PDF following the instructions in [Generating PDF](#generating-pdf)

### Generating new URDF files

First, make sure to install ROS and the corresponding repository that comes along with the URDF file (which I'll call the URDF repository). Build the repository using ROS, and make sure it builds without errors. You may need to install more dependencies, depending on what the URDF repository says.

Next, find the xacro file used to create the URDF. This will most likely be found in the URDF repository. Manually comment out any non-kinematic structures. Then run the following command to generate the new URDF file:

```
rosrun xacro xacro <xacro_path> > <new_urdf_path>
```

### Generating PDF
To generate the pdf file that corresponds to the urdf file, install ROS 1, and run the following command:

```
sudo apt-get install graphviz
```

Next, follow the instructions here: https://manpages.ubuntu.com/manpages/jammy/man1/urdf_to_graphiz.1.html

