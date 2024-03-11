import numpy as np
import pandas as pd
import csv


class Robot_dataset:

    def __init__(self, data_path):
        self.data = []
        self.data = pd.read_csv(data_path, header=None)
        print(self.data.head())
        # print(self.data)
        
        # catagorize dataset
        self.time = self.data.to_numpy()[:,0:2]
        time_col = {'time','real_time','TIME_MOTOR_r'}
        self.time.rename(time_col)

        self.position = self.data.to_numpy()[:,3:14]
        position_col = ['LF_HAA_q', 'LF_HFE_q','LF_KFE_q','RF_HAA_q', 'RF_HFE_q','RF_KFE_q',
                        'LH_HAA_q', 'LH_HFE_q','LH_KFE_q','RH_HAA_q', 'RH_HFE_q','RH_KFE_q']
        self.position.rename(position_col)

        self.joint_speed = self.data.to_numpy()[:,15:26]
        joint_speed_col = ['LF_HAA_qd_f', 'LF_HFE_qd_f','LF_KFE_qd_f','RF_HAA_qd_f', 'RF_HFE_qd_f','RF_KFE_qd_f',
                           'LH_HAA_qd_f', 'LH_HFE_qd_f','LH_KFE_qd_f','RH_HAA_qd_f', 'RH_HFE_qd_f','RH_KFE_qd_f']
        self.position.rename(joint_speed_col)    

        self.torque = self.data.to_numpy()[:.27:38]
        torque_col = ['LF_HAA_fm', 'LF_HFE_fm','LF_KFE_fm','RF_HAA_fm', 'RF_HFE_fm','RF_KFE_fm',
                      'LH_HAA_fm', 'LH_HFE_fm','LH_KFE_fm','RH_HAA_fm', 'RH_HFE_fm','RH_KFE_fm']
        self.position.rename(torque_col)

        self.IMU = self.data.to_numpy()[:,39:54]
        IMU_col = ['B_Q_0_r', 'B_Q_1_r','B_Q_2_r','B_Q_3_r',
                   'B_Ad_A_r', 'B_Ad_B_r', 'B_Ad_G_r',
                   'B_Xdd_r', 'B_Ydd_r','B_Zdd_r',
                   'B2_Ad_A_r', 'B2_Ad_B_r', 'B2_Ad_G_r',
                   'B2_Xdd_r', 'B2_Ydd_r','B2_Zdd_r']
        self.position.rename(IMU_col)

        self.relative_encoder = self.data.to_numpy()[:,55:66]
        rel_encoder_col = ['LF_HAA_rel_enc', 'LF_HFE_rel_enc','LF_KFE_rel_enc','RF_HAA_rel_enc', 'RF_HFE_rel_enc','RF_KFE_rel_enc',
                           'LH_HAA_rel_enc', 'LH_HFE_rel_enc','LH_KFE_rel_enc','RH_HAA_rel_enc', 'RH_HFE_rel_enc','RH_KFE_rel_enc']
        self.position.rename(rel_encoder_col)

        self.absolute_encoder = self.data.to_numpy()[:,67:78]
        abs_encoder_col = ['LF_HAA_abs_enc', 'LF_HFE_abs_enc','LF_KFE_abs_enc','RF_HAA_abs_enc', 'RF_HFE_abs_enc','RF_KFE_abs_enc',
                           'LH_HAA_abs_enc', 'LH_HFE_abs_enc','LH_KFE_abs_enc','RH_HAA_abs_enc', 'RH_HFE_abs_enc','RH_KFE_abs_enc']
        self.position.rename(abs_encoder_col)

        self.IMU_time_stamp = self.data.to_numpy()[:,79:-1]
        IMU_time_stamp_col = {'imu_time_stamp1','imu_time_stamp2'}
        self.time.rename(IMU_time_stamp_col)

        return self.data



path = "D:\State estimation\dataset code\sensors.csv"
data = Robot_dataset(path)
print(data)

        
    
# def main():
#     path = "D:\State estimation\dataset code\sensors.csv"
#     data = Robot_dataset(path)

#     print(data[:,0])

    

        