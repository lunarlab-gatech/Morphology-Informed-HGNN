from pathlib import Path
import numpy as np
import pinocchio as pin
from mi_hgnn import QuadSDKDataset_A1Speed0_5, QuadSDKDataset_A1Speed1_0, QuadSDKDataset_A1Speed1_5FlippedOver
from os.path import dirname, join, abspath
import matplotlib.pyplot as plt

np.set_printoptions(linewidth=np.inf)

## 0. DATA
pinocchio_model_dir = join(dirname(dirname(str(abspath(__file__)))), "urdf_files")

model_path = join(pinocchio_model_dir, "A1")
mesh_dir = pinocchio_model_dir
urdf_filename = "a1_updated.urdf"
urdf_model_path = join(model_path, urdf_filename)

model, collision_model, visual_model = pin.buildModelsFromUrdf(
    urdf_model_path, mesh_dir, pin.JointModelFreeFlyer()
)
data = model.createData()
print(model)

## FIND CONTACTS
## First, we set the frame for our contacts. We assume the contacts are placed at the following 4 frames and they are 3D
feet_names = ["FL_foot", "FR_foot", "RL_foot", "RR_foot"]
feet_ids = [model.getFrameId(n) for n in feet_names]
ncontact = len(feet_names)

# v0 = np.zeros(model.nv)
a0 = np.zeros(model.nv) # set acceleration for 0 for drift calculation

## Get data
## Define model type
history_length = 1000
model_type = 'heterogeneous_gnn'
path_to_urdf = Path('urdf_files', 'A1', 'a1.urdf').absolute()
path_to_quad_sdk_1 = Path(
            Path('.').parent, 'datasets', 'QuadSDK-A1Speed1.0').absolute()
dataset_1 = QuadSDKDataset_A1Speed1_0(
        path_to_quad_sdk_1, path_to_urdf, 'package://a1_description/',
        'unitree_ros/robots/a1_description', model_type, history_length)

quad_data = dataset_1.load_data_sorted(0)

print(quad_data[10])

joint_timestamp = quad_data[10][:,1]
imu_timestamp = quad_data[10][:,2]

# load_data_sorted returns info so that it matches order of the URDF file (FR, FL, RR, RL)
# However, pinocchio has order of FL, FR, RL, RR
foot_node_indices_sorted = [3, 4, 5, 0, 1, 2, 9, 10, 11, 6, 7, 8]

lin_acc = quad_data[0]     # IMU linear acc
ang_vel = quad_data[1]     # IMU angular vel
j_p = quad_data[2][:, foot_node_indices_sorted]         # joint position 
j_v = quad_data[3][:, foot_node_indices_sorted]         # joint velocity             
j_T = quad_data[4][:, foot_node_indices_sorted]         # joint torque   
# labels = quad_data[7]    # Dataset labels (Ground Truth GRF)

# Set Robot Position and Orientation to World Frame
#r_p = np.zeros((1000, 3))                                  # quad_data[8]         # Robot position (x, y, z)
#r_o = np.hstack((np.zeros((1000, 3)), np.ones((1000, 1)))) # quad_data[9]         # Robot orientation (x, y, z, w) quaternion
r_p = quad_data[8]
r_o = quad_data[9]
q = np.hstack((np.hstack((r_p, r_o)), j_p))     # state:[x, y, z, quaternions, 4*(Hip_joint, Thigh_joint, Calf_joint)position]
q = q[1:-1, :]

## Data Preprocessing and arrangement
lin_vel, ang_acc, j_acc = np.zeros((0, 3)), np.zeros((0, 3)), np.zeros((0, 12))

## Taking derivative to get lin_vel, ang_acc, and j_acc
for i in range (1, len(r_p)-1):
    new_lin_vel = (r_p[i+1, :] - r_p[i-1, :]) / (imu_timestamp[i + 1] - imu_timestamp[i - 1])
    lin_vel = np.vstack((lin_vel, new_lin_vel))

    new_ang_acc = (ang_vel[i+1, :] - ang_vel[i-1, :]) / (imu_timestamp[i + 1] - imu_timestamp[i - 1])
    ang_acc = np.vstack((ang_acc, new_ang_acc))

    new_j_acc = (j_v[i+1, :] - j_v[i-1, :]) / (joint_timestamp[i + 1] - joint_timestamp[i - 1])
    j_acc = np.vstack((j_acc, new_j_acc))

vel = np.hstack(( np.hstack((lin_vel[:, :], ang_vel[1:-1, :])), j_v[1:-1, :]))           # velocity term
acc = np.hstack(( np.hstack((lin_acc[1:-1, :], ang_acc[:, :])), j_acc[:, :]))           # acceleration term
tau = np.hstack((np.zeros([len(vel), 6]), j_T[1:-1, :]))                                  # torque term

real_vel = np.zeros([len(vel), 18])
real_acc = np.zeros([len(acc), 18])
f_c = 2.0
alpha = 1/(2*np.pi*f_c)
for i in range (1,len(vel)):
    real_vel[i,:] = alpha * vel[i,:] + (1-alpha) * real_vel[i-1,:]
    real_acc[i,:] = alpha * acc[i,:] + (1-alpha) * real_acc[i-1,:]

FL_GRF_cal = np.zeros([len(vel), 3])  # GRF of FL_FOOT (Local Frame)
FR_GRF_cal = np.zeros([len(vel), 3])  # GRF of FR_FOOT (Local Frame)
RL_GRF_cal = np.zeros([len(vel), 3])  # GRF of HL_FOOT (Local Frame)
RR_GRF_cal = np.zeros([len(vel), 3])  # GRF of HR_FOOT (Local Frame)

for i in range (len(vel)):
    # Find Mass matrix
    M = pin.crba(model, data, q[i,:])

    # Compute dynamic drift -- Coriolis, centrifugal, gravity
    drift = pin.rnea(model, data, q[i,:], vel[i,:], a0)

    # Now, we need to find the contact Jacobians.
    # These are the Jacobians that relate the joint velocity to the velocity of each feet
    J_feet = [np.copy(pin.computeFrameJacobian(model, data, q[i,:], id, pin.LOCAL)) for id in feet_ids]
    J_feet_first_3_rows = [np.copy(J[:3, :]) for J in J_feet] 
    J_feet_T = np.zeros([18, 3 * ncontact])
    J_feet_T[:, :] = np.vstack(J_feet_first_3_rows).T

    # Contact forces at local coordinates (at each foot coordinate)
    contact_forces = np.linalg.pinv(J_feet_T) @ ((M @ acc[i,:]) + drift - tau[i, :])
    contact_forces_split = np.split(contact_forces, ncontact)

    # Compute the placement of each frame
    pin.framesForwardKinematics(model, data, q[i,:])

    # Convert Contact forces to World frame
    world_frame_forces = []
    world_id = model.getFrameId("universe")
    for force, foot_id in zip(contact_forces_split, feet_ids):
        force_transpose = np.array([[force[0]], [force[1]], [force[2]]])

        # Get the Foot to World Transform
        world_to_foot_SE3 = data.oMf[foot_id]
        foot_to_world_SE3 = world_to_foot_SE3.inverse()

        # Transform the force into the world frame
        world_frame_force = foot_to_world_SE3.rotation @ force_transpose
        world_frame_forces.append(np.copy(world_frame_force.squeeze()))

    # Extract and save the forces for this timestep
    FL_GRF_cal[i] = world_frame_forces[0]
    FR_GRF_cal[i] = world_frame_forces[1]
    RL_GRF_cal[i] = world_frame_forces[2]
    RR_GRF_cal[i] = world_frame_forces[3]

desired_length = 1000
history_length = 1
model_type = 'heterogeneous_gnn'
path_to_urdf = Path('urdf_files', 'A1', 'a1.urdf').absolute()
path_to_quad_sdk_1 = Path(
                Path('.').parent, 'datasets', 'QuadSDK-A1Speed1.0').absolute()
dataset_1 = QuadSDKDataset_A1Speed1_0(
            path_to_quad_sdk_1, path_to_urdf, 'package://a1_description/',
            'unitree_ros/robots/a1_description', model_type, history_length)
    

# load_data_sorted returns info so that it matches order of the URDF file (FR, FL, RR, RL)
labels = np.zeros([desired_length, 4])
for i in range (desired_length):
        quad_data = dataset_1.load_data_sorted(i)
        labels[i] = quad_data[7]      # Dataset labels (Groud Truth GRF)
 
# Plot FL_FOOT
plt.title('FL_FOOT_GRF')
plt.plot(FR_GRF_cal[:,2])
plt.plot(labels[:,0])
plt.ylabel('Z_direction')
plt.legend(["Model_Based","Ground Truth"])
plt.show()

# Plot FR_FOOT
plt.title('FR_FOOT_GRF')
plt.plot(FL_GRF_cal[:,2])
plt.plot(labels[:,1])
plt.ylabel('Z_direction')
plt.legend(["Model_Based","Ground Truth"])
plt.show()

# Plot HL_Foot
plt.title('HL_FOOT_GRF')
plt.plot(RR_GRF_cal[:,2])
plt.plot(labels[:,2])
plt.ylabel('Z_direction')
plt.legend(["Model_Based","Ground Truth"])
plt.show()

# Plot HR_FOOT
plt.title('HR_FOOT')
plt.plot(RL_GRF_cal[:,2])
plt.plot(labels[:,3])
plt.ylabel('Z_direction')
plt.legend(["Model_Based","Ground Truth"])
plt.show()