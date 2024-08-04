from pathlib import Path
import numpy as np
import pinocchio as pin
from grfgnn import QuadSDKDataset_A1Speed0_5, QuadSDKDataset_A1Speed1_0, QuadSDKDataset_A1Speed1_5FlippedOver
from os.path import dirname, join, abspath
import matplotlib.pyplot as plt

np.set_printoptions(linewidth=np.inf)

# ----- PROBLEM STATEMENT ------
#
# We want to find the contact forces and torques required to stand still at a configuration 'q0'.
# We assume 3D contacts at each of the feet
#
# The dynamic equation would look like:
#
# M*q_ddot + g(q) + C(q, q_dot) = tau + J^T*lambda --> (for the static case) --> g(q) = tau + Jc^T*lambda (1)

# ----- SOLVING STRATEGY ------

# Split the equation between the base link (_bl) joint and the rest of the joints (_j). That is,
#
#  | g_bl |   |  0  |   | Jc__feet_bl.T |   | l1 |
#  | g_j  | = | tau | + | Jc__feet_j.T  | * | l2 |    (2)
#                                           | l3 |
#                                           | l4 |

# First, find the contact forces l1, l2, l3, l4 (these are 3 dimensional) by solving for the first 6 rows of (2).
# That is,
#
# g_bl   = Jc__feet_bl.T * | l1 |
#                          | l2 |    (3)
#                          | l3 |
#                          | l4 |
#
# Thus we find the contact froces by computing the jacobian pseudoinverse,
#
# | l1 | = pinv(Jc__feet_bl.T) * g_bl  (4)
# | l2 |
# | l3 |
# | l4 |
#
# Now, we can find the necessary torques using the bottom rows in (2). That is,
#
#                             | l1 |
#  tau = g_j - Jc__feet_j.T * | l2 |    (5)
#                             | l3 |
#                             | l4 |

# ----- SOLUTION ------

## 0. DATA
pinocchio_model_dir = join(dirname(dirname(str(abspath(__file__)))), "urdf_files")

model_path = join(pinocchio_model_dir, "A1")
mesh_dir = pinocchio_model_dir
urdf_filename = "a1_updated.urdf"
urdf_model_path = join(model_path, urdf_filename)

model, collision_model, visual_model = pin.buildModelsFromUrdf(
    urdf_model_path, mesh_dir, pin.JointModelFreeFlyer(), verbose= True
)

data = model.createData()
print(model)
## FIND CONTACTS
## First, we set the frame for our contacts. We assume the contacts are placed at the following 4 frames and they are 3D
feet_names = ["FL_foot", "FR_foot", "RL_foot", "RR_foot"]
feet_ids = [model.getFrameId(n) for n in feet_names]
# print(feet_ids)
bl_id = model.getFrameId("base")
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

quad_data = dataset_1.load_data_sorted_with_timestamps(0)

delta_T = quad_data[10]    # Time stamp
joint_timestamp = delta_T[0][1]
imu_timestamp = delta_T[0][2]
# print(joint_timestamp)

lin_acc = quad_data[0]     # IMU linear acc
# print(lin_acc[1:-1, :])
ang_vel = quad_data[1]     # IMU angular vel

foot_node_indices_sorted = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]

new_j_p = quad_data[2]         # joint position 
j_p = new_j_p[:,foot_node_indices_sorted]
# print(j_p)               
new_j_v = quad_data[3]         # joint velocity 
j_v = new_j_v[:,foot_node_indices_sorted]
# print(j_v)               
j_T = quad_data[4]         # joint torque   
f_p = quad_data[5]         # Foot position
f_v = quad_data[6]         # Foot velocity
# labels = quad_data[7]    # Dataset labels (Groud Truth GRF)
r_p = quad_data[8]         # Robot position (x, y, z)
# print(r_p)
r_o = quad_data[9]         # Robot orientation (x, y, z, w) quaternion
# print(r_o)
q = np.hstack((np.hstack((r_p, r_o)), j_p))     # state:[x, y, z, quaternions, 4*(Hip_joint, Thigh_joint, Calf_joint) position]
q = q[1:-1, :]
print(q[0,:])
# Js__feet_q = [
#         np.copy(pin.computeFrameJacobian(model, data, q[0,:], id, pin.LOCAL)) for id in feet_ids
#     ]
# print(Js__feet_q)
## Data Preprocessing and arrangement
lin_vel, ang_acc, j_acc = [[0,0,0]], [[0,0,0]], [[0,0,0,0,0,0,0,0,0,0,0,0]]

## Taking derivative to get lin_vel, ang_acc, and j_acc
for i in range (1, len(r_p)-1):
    new_lin_vel = (r_p[i+1, :] - r_p[i-1, :]) / (2 * imu_timestamp)
    lin_vel = np.vstack((lin_vel, new_lin_vel))

    new_ang_acc = (ang_vel[i+1, :] - ang_vel[i-1, :]) / (2 * imu_timestamp)
    ang_acc = np.vstack((ang_acc, new_ang_acc))

    new_j_acc = (j_v[i+1, :] - j_v[i-1, :]) / (2 * joint_timestamp)
    j_acc = np.vstack((j_acc, new_j_acc))

vel = np.hstack(( np.hstack((lin_vel[1:, :], ang_vel[1:-1, :])), j_v[1:-1, :]))           # velocity term
acc = np.hstack(( np.hstack((lin_acc[1:-1, :], ang_acc[1:, :])), j_acc[1:, :]))           # acceleration term
tau = np.hstack((np.zeros([len(vel), 6]), j_T[1:-1, :]))                                  # torque term
# print(len(tau[0]))

FL_GRF_cal = np.zeros([len(vel), 3])  # GRF of FL_FOOT (Local Frame)
FR_GRF_cal = np.zeros([len(vel), 3])  # GRF of FR_FOOT (Local Frame)
HL_GRF_cal = np.zeros([len(vel), 3])  # GRF of HL_FOOT (Local Frame)
HR_GRF_cal = np.zeros([len(vel), 3])  # GRF of HR_FOOT (Local Frame)

FL_torque = np.zeros([len(vel), 3])  # Torque of FL_FOOT 
FR_torque = np.zeros([len(vel), 3])  # Torque of FR_FOOT 
HL_torque = np.zeros([len(vel), 3])  # Torque of HL_FOOT 
HR_torque = np.zeros([len(vel), 3])  # Torque of HR_FOOT 

for i in range (len(vel)):
    # Find Mass matrix and Drift
    # compute mass matrix M
    M = pin.crba(model, data, q[i,:])
    acc_term = M @ acc[i,:]

    acc_bl = acc_term[:6]
    acc_j = acc_term[6:]
    # compute dynamic drift -- Coriolis, centrifugal, gravity
    drift = pin.rnea(model, data, q[i,:], vel[i,:], a0)

    b_bl = drift[:6]
    b_j = drift[6:]

    # Now, we need to find the contact Jacobians appearing in (1).
    # These are the Jacobians that relate the joint velocity  to the velocity of each feet
    Js__feet_q = [
        np.copy(pin.computeFrameJacobian(model, data, q[i,:], id, pin.LOCAL)) for id in feet_ids
    ]
    # print(Js__feet_q)

    Js__feet_bl = [np.copy(J[:3, :6]) for J in Js__feet_q]

    # Notice that we can write the equation above as an horizontal stack of Jacobians transposed and vertical stack of contact forces
    Jc__feet_bl_T = np.zeros([6, 3 * ncontact])
    Jc__feet_bl_T[:, :] = np.vstack(Js__feet_bl).T
    # print(Js__feet_bl)

    # Now I only need to do the pinv to compute the contact forces

    ls = np.linalg.pinv(Jc__feet_bl_T) @ (acc_bl + b_bl)  # This is (3)

    # Contact forces at local coordinates (at each foot coordinate)
    ls__f = np.split(ls, ncontact)
    
    FL_GRF_cal[i] = ls__f[0]
    FR_GRF_cal[i] = ls__f[1]
    HL_GRF_cal[i] = ls__f[2]
    HR_GRF_cal[i] = ls__f[3]

    pin.framesForwardKinematics(model, data, q[i,:])

    # Contact forces at base link frame
    # ls__bl = []
    # for l__f, foot_id in zip(ls__f, feet_ids):
    #     l_sp__f = pin.Force(l__f, np.zeros(3))
    #     l_sp__bl = data.oMf[bl_id].actInv(data.oMf[foot_id].act(l_sp__f))
    #     ls__bl.append(np.copy(l_sp__bl.vector))

    # 3. FIND TAU
    # Find Jc__feet_j
    Js_feet_j = [np.copy(J[:3, 6:]) for J in Js__feet_q]

    Jc__feet_j_T = np.zeros([12, 3 * ncontact])
    Jc__feet_j_T[:, :] = np.vstack(Js_feet_j).T

    # Apply (5)
    tau = (acc_j + b_j) - (Jc__feet_j_T @ ls)

    FL_torque[i] = tau[0:3]
    FR_torque[i] = tau[3:6]
    HL_torque[i] = tau[6:9]
    HR_torque[i] = tau[9:12]
    # print(FL_torque)

# 4. CROSS CHECKS

# # INVERSE DYNAMICS
# # We can compare this torques with the ones one would obtain when computing the ID considering the external forces in ls
# pin.framesForwardKinematics(model, data, q0)

# joint_names = ["FL_KFE", "FR_KFE", "HL_KFE", "HR_KFE"]
# joint_ids = [model.getJointId(n) for n in joint_names]

# fs_ext = [pin.Force(np.zeros(6)) for _ in range(len(model.joints))]
# for idx, joint in enumerate(model.joints):
#     if joint.id in joint_ids:
#         fext__bl = pin.Force(ls__bl[joint_ids.index(joint.id)])
#         fs_ext[idx] = data.oMi[joint.id].actInv(data.oMf[bl_id].act(fext__bl))

# tau_rnea = pin.rnea(model, data, q0, v0, a0, fs_ext)

# print("\n--- ID: JOINT TORQUES ---")
# print("Tau from RNEA:         {}".format(tau_rnea))
# print("Tau computed manually: {}".format(np.append(np.zeros(6), tau)))
# print("Tau error: {}".format(np.linalg.norm(np.append(np.zeros(6), tau) - tau_rnea)))

# # FORWARD DYNAMICS
# # We can also check the results using FD. FD with the tau we got, q0 and v0, should give 0 acceleration and the contact forces
# Js_feet3d_q = [np.copy(J[:3, :]) for J in Js__feet_q]
# acc = pin.forwardDynamics(
#     model,
#     data,
#     q0,
#     v0,
#     np.append(np.zeros(6), tau),
#     np.vstack(Js_feet3d_q),
#     np.zeros(12),
# )

# print("\n--- FD: ACC. & CONTACT FORCES ---")
# print("Norm of the FD acceleration: {}".format(np.linalg.norm(acc)))
# print("Contact forces manually: {}".format(ls))
# print("Contact forces FD: {}".format(data.lambda_c))
# print("Contact forces error: {}".format(np.linalg.norm(data.lambda_c - ls)))


desired_length = 1000
history_length = 1
model_type = 'heterogeneous_gnn'
path_to_urdf = Path('urdf_files', 'A1', 'a1.urdf').absolute()
path_to_quad_sdk_1 = Path(
                Path('.').parent, 'datasets', 'QuadSDK-A1Speed1.0').absolute()
dataset_1 = QuadSDKDataset_A1Speed1_0(
            path_to_quad_sdk_1, path_to_urdf, 'package://a1_description/',
            'unitree_ros/robots/a1_description', model_type, history_length)
    
labels = np.zeros([desired_length, 4])
for i in range (desired_length):
        quad_data = dataset_1.load_data_sorted_with_timestamps(i)
        labels[i] = quad_data[7]      # Dataset labels (Groud Truth GRF)
        
# Plot FL_FOOT
plt.title('FL_FOOT_GRF')
plt.plot(FL_GRF_cal[:,2])
plt.plot(labels[:,0])
plt.ylabel('Z_direction')
plt.legend(["Model_Based","Ground Truth"])
plt.show()

# Plot FR_FOOT
plt.title('FR_FOOT_GRF')
plt.plot(FR_GRF_cal[:,2])
plt.plot(labels[:,1])
plt.ylabel('Z_direction')
plt.legend(["Model_Based","Ground Truth"])
plt.show()

# Plot HL_Foot
plt.title('HL_FOOT_GRF')
plt.plot(HL_GRF_cal[:,2])
plt.plot(labels[:,2])
plt.ylabel('Z_direction')
plt.legend(["Model_Based","Ground Truth"])
plt.show()

# Plot HR_FOOT
plt.title('HR_FOOT')
plt.plot(HR_GRF_cal[:,2])
plt.plot(labels[:,3])
plt.ylabel('Z_direction')
plt.legend(["Model_Based","Ground Truth"])
plt.show()

# def get_label(desired_length, history_length):
#     # history_length = 1
#     model_type = 'heterogeneous_gnn'
#     path_to_urdf = Path('urdf_files', 'A1', 'a1.urdf').absolute()
#     path_to_quad_sdk_1 = Path(
#                 Path('.').parent, 'datasets', 'QuadSDK-A1Speed1.0').absolute()
#     dataset_1 = QuadSDKDataset_A1Speed1_0(
#             path_to_quad_sdk_1, path_to_urdf, 'package://a1_description/',
#             'unitree_ros/robots/a1_description', model_type, history_length)
    
#     labels = np.zeros([desired_length, 4])
#     for i in range (desired_length):
#         quad_data = dataset_1.load_data_sorted_with_timestamps(i)
#         labels[i] = quad_data[7]      # Dataset labels (Groud Truth GRF)

#     return labels