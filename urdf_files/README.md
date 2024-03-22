# URDF files

Each robot URDF file with its corresponding repository can be found in a folder with its name.

Note, the ```urdfParser.py``` file will take the ```*.urdf``` file and generate a ```*_updated.urdf``` file in the same directory, which contains updated paths based on your current system. For this reason, these files are not commited to the GitHub repo, as they should vary per device.

## Generating PDF
To generate the pdf file that corresponds to the urdf file, install ROS 1, and run the following command:
```
sudo apt-get install graphviz
```

Next, follow the instructions here: https://manpages.ubuntu.com/manpages/jammy/man1/urdf_to_graphiz.1.html

