# Implementing Custom Datasets

We hope that many people use our MI-HGNN on a variety of datasets. We provide the `FlexibleDataset` class which provides many convenient features and can be inherited for use with custom datasets. Below is a short summary of its features:
- Automatic download of relevant datasets from the Internet (from Google Drive or Dropbox).
- Data sorting to match the order of joint, foot, and base nodes in the robot graph.
- Wrapper for the `RobotGraph` class that generates the graph from the robot URDF file.
- Easy customization with custom history lengths and a normalization parameter.
- Provides custom get() function returns for training both an MLP and the MI-HGNN.
- Option for easy evaluation on floating-base dynamics model, though our current implementation is specific for the simulated A1 robot in our paper, meaning changes will be necessary for proper results on your robot.

However, `FlexibleDataset` currently only supports the following input data:
- lin_acc (np.array) - IMU linear acceleration
- ang_vel (np.array) - IMU angular velocity
- j_p (np.array) - Joint positions
- j_v (np.array) - Joint velocities
- j_T (np.array) - Joint Torques
- f_p (np.array) - Foot position
- f_v (np.array) - Foot velocity
- labels (np.array) - The Dataset labels (either Z direction GRFs, or contact states)
- r_p (np.array) - Robot position (GT)
- r_o (np.array) - Robot orientation (GT) as a quaternion, in the order (x, y, z, w)
- timestamps (np.array) - Array containing the timestamps of the data

Also note that not all of these are used depending on the applied model (MLP vs. MIHGNN vs Floating-Base Dynamics).

If `FlexibleDataset` supports your input data, then you can easily use it by writing a simple dataset class that inherits from `FlexibleDataset`, similar to `LinTzuYaunDataset` or `QuadSDKDataset`. We've provided a template for you in the `CustomDatasetTemplate.py` file, which you can use to start. 

## Using the Custom Dataset Template

This section will explain how to edit the `CustomDatasetTemplate.py` file for use with your own dataset to take advantage of the features of the `FlexibleDataset` class. 

First, open the file and rename the class to your liking.

### Adding Dataset Sequences
Next, scroll down to the bottom of the file where it says `DATASET SEQUENCES`. Add every sequence of your dataset as its own class, which will require you to upload the data either to Dropbox or Google. See `CustomDatasetTemplate.py` for details.

This is a clean way for data loading, as it allows the user to later combine different sequences as they'd like with the `torch.utils.data.ConcatDataset` function (see `research/train_classification_sample_eff.py` for an example). Defining these classes also means that training an MI-HGNN model on a different computer doesn't require the user to manually download any datasets, as `FlexibleDataset` will do it for you.

Also, when the files are downloaded, they will be renamed to the value provided by `get_downloaded_dataset_file_name()`. Overwrite this function so that the file extension is correct (`.mat` for a Matlab file, `.bag` for a ROSbag file, etc).

### Implementing Data Processing
Now that you can load your dataset files, you need to implement processing. This step should be implemented in `process()`, and should convert the file from whatever format it is currently in into a `.mat` file for fast training speeds. You'll also need to provide code for extracting the number of dataset entries in this sequence, which will be saved into a .txt file for future use.

Implement this function. You can see `quadSDKDataset.py` for an example of converting a ROSbag file into a .mat file.

### Implementing Data Loading
Now that data is loaded and processed, you can now implement the function for opening the .mat file and extracting the relevant dataset sequence.
This should be done in `load_data_at_dataset_seq()`. The .mat file you saved in the last step will now be available at `self.mat_data` for easy access.
Note that this function will also need to use the `self.history_length` parameter to support training with a history of measurements. See `CustomDatasetTemplate.py` for details, and see `LinTzuYaunDataset.py` for a proper implementation of this function.

### Setting the proper URDF file
Since its easy for the user to provide the wrong URDF file for a dataset sequence, `FlexibleDataset` checks that the URDF file provided by the user matches what the dataset expects. You can tell `FlexibleDataset` which URDF file should be used with this dataset by going to the URDF file and copying the name found at the top of the file, like pictured below:

```
<robot name="miniCheetah">
```

This name should be pasted into `get_expected_urdf_name()`.

### Facilitating Data Sorting
Finally, the last step is to tell `FlexibleDataset` what order your dataset data is in. For example, which index in the joint position array corresponds to a specific joint in the URDF file? To do this, you'll implement `get_urdf_name_to_dataset_array_index()`. See `CustomDatasetTemplate.py` for more details.

After doing this, your dataset will work with our current codebase for training MLP and MI-HGNN models! You can now instantiate your dataset and use it like in the examples in the `research` directory. Happy Training!
