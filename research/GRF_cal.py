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
    urdf_model_path, mesh_dir, pin.JointModelFreeFlyer()
)

data = model.createData()

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
j_p = quad_data[2]         # joint position 
# print(j_p)
j_v = quad_data[3]         # joint velocity 
# print(j_v)               
j_T = quad_data[4]         # joint torque   
f_p = quad_data[5]         # Foot position
f_v = quad_data[6]         # Foot velocity
# labels = quad_data[7]    # Dataset labels (Groud Truth GRF)
r_p = quad_data[8]         # Robot position (x, y, z)
# print(r_p)
r_o = quad_data[9]         # Robot orientation (x, y, z, w) quaternion
# print(r_o)
q = np.hstack((np.hstack((r_p, r_o)), j_p))     # state:[x, y, z, quaternions, 4*(Hip_joint, Thigh_joint, Calf_joint)position]
q = q[1:-1, :]
# print(len(q))

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

for i in range (len(vel)):
    # Find Mass matrix and Drift
    # compute mass matrix M
    M = pin.crba(model, data, q[i,:])

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

    # Find Jc__feet_j
    Js_feet_j = [np.copy(J[:3, 6:]) for J in Js__feet_q]

    Jc__feet_j_T = np.zeros([12, 3 * ncontact])
    Jc__feet_j_T[:, :] = np.vstack(Js_feet_j).T

    # Combine Jc__feet_bl_T & Jc__feet_j_T
    Jc__feet_T = np.vstack((Jc__feet_bl_T, Jc__feet_j_T))
    # print(Jc__feet_T)

    # Now I only need to do the pinv to compute the contact forces

    # ls = np.linalg.pinv(Jc__feet_T) @ (M @ acc[i,:] + drift - tau[i, :])  # This is (3)
    ls = np.dot(np.linalg.pinv(Jc__feet_T), (np.dot(M, acc[i,:]) + drift - tau[i, :]))  # This is (3)

    # Contact forces at local coordinates (at each foot coordinate)
    ls__f = np.split(ls, ncontact)
    
    FL_GRF_cal[i] = ls__f[0]
    FR_GRF_cal[i] = ls__f[1]
    HL_GRF_cal[i] = ls__f[2]
    HR_GRF_cal[i] = ls__f[3]

    pin.framesForwardKinematics(model, data, q[i,:])

    # # Contact forces at base link frame
    ls__bl = []
    for l__f, foot_id in zip(ls__f, feet_ids):
        l_sp__f = pin.Force(l__f, np.zeros(3))
        l_sp__bl = data.oMf[bl_id].actInv(data.oMf[foot_id].act(l_sp__f))
        ls__bl.append(np.copy(l_sp__bl.vector))

    # print(l_sp__bl)

    # print("\n--- CONTACT FORCES ---")
    # for l__f, foot_id, name in zip(ls__bl, feet_ids, feet_names):
    #     print("Contact force at foot {} expressed at the BL is: {}".format(name, l__f))

    # # Notice that if we add all the contact forces are equal to the g_grav
    # print(
    #     "Error between contact forces and gravity at base link: {}".format(
    #         np.linalg.norm(b_bl - sum(ls__bl))
    #     )
    # )

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