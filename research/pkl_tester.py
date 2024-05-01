import pickle

with open('datasets/go1_dataset_actual_freq/raw/traj_0003.pkl', 'rb') as f:
    # Dict with following keys
    # 'x_vel', 'y_vel', 'z_vel', 'r_ang', 'p_ang', 'y_ang', 'joint_positions', 'joint_velocities'
    data: dict = pickle.load(f)
    print(type(data))
    print(data.keys())
    print(type(data["joint_velocities"][0][0]))