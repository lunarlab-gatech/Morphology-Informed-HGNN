import pandas as pd
import os
    
class RobotCSV:
    def __init__(self, dataset_name):
        # Create the system path to the data
        data_path = os.path.join(os.getcwd(), "datasets", dataset_name, dataset_name, "data", "sensors.csv")

        # Read in the data
        self.data = pd.read_csv(data_path, header=None)
        self.data = self.data.to_numpy()

        # catagorize dataset
        columns = ['time' ,'real_time','TIME_MOTOR_r',
                   'LF_HAA_q', 'LF_HFE_q','LF_KFE_q','RF_HAA_q', 'RF_HFE_q','RF_KFE_q',
                   'LH_HAA_q', 'LH_HFE_q','LH_KFE_q','RH_HAA_q', 'RH_HFE_q','RH_KFE_q',
                   'LF_HAA_qd_f', 'LF_HFE_qd_f','LF_KFE_qd_f','RF_HAA_qd_f', 'RF_HFE_qd_f','RF_KFE_qd_f',
                   'LH_HAA_qd_f', 'LH_HFE_qd_f','LH_KFE_qd_f','RH_HAA_qd_f', 'RH_HFE_qd_f','RH_KFE_qd_f',
                   'LF_HAA_fm', 'LF_HFE_fm','LF_KFE_fm','RF_HAA_fm', 'RF_HFE_fm','RF_KFE_fm',
                   'LH_HAA_fm', 'LH_HFE_fm','LH_KFE_fm','RH_HAA_fm', 'RH_HFE_fm','RH_KFE_fm',
                   'B_Q_0_r', 'B_Q_1_r','B_Q_2_r','B_Q_3_r',
                   'B_Ad_A_r', 'B_Ad_B_r', 'B_Ad_G_r',
                   'B_Xdd_r', 'B_Ydd_r','B_Zdd_r',
                   'B2_Ad_A_r', 'B2_Ad_B_r', 'B2_Ad_G_r',
                   'B2_Xdd_r', 'B2_Ydd_r','B2_Zdd_r',
                   'LF_HAA_rel_enc', 'LF_HFE_rel_enc','LF_KFE_rel_enc','RF_HAA_rel_enc', 'RF_HFE_rel_enc','RF_KFE_rel_enc',
                   'LH_HAA_rel_enc', 'LH_HFE_rel_enc','LH_KFE_rel_enc','RH_HAA_rel_enc', 'RH_HFE_rel_enc','RH_KFE_rel_enc',
                   'LF_HAA_abs_enc', 'LF_HFE_abs_enc','LF_KFE_abs_enc','RF_HAA_abs_enc', 'RF_HFE_abs_enc','RF_KFE_abs_enc',
                   'LH_HAA_abs_enc', 'LH_HFE_abs_enc','LH_KFE_abs_enc','RH_HAA_abs_enc', 'RH_HFE_abs_enc','RH_KFE_abs_enc',
                   'imu_time_stamp1','imu_time_stamp2']
        
        # Create a dictionary to store the results
        self.data_dict = {}
        for i in range (0, len(columns)):
            self.data_dict[columns[i]] = self.data[:,i]

    def pull_values(self, variable_name):
        return self.data_dict[variable_name]
    
    # Return the number of enteries in the dataset
    def num_dataset_entries(self):
        return len(self.data_dict['time'])


def main():
    dataset_name = "trot_in_lab_1"
    data = RobotCSV(dataset_name)
    print("Time: ", data.pull_values('time'))
    print("Left front hip a/a joint position: ", data.pull_values('LF_HAA_q'))

if __name__ == "__main__":
    main()


    

        