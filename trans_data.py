import os
import numpy as np

data_list = ["/home/flexiv/Code/FLEXIV/flexiv_sw/flexiv/lib_py/experimental/hole_axis/data/Pin_1/2021_07_05_19:28:26",
             "/home/flexiv/Code/FLEXIV/flexiv_sw/flexiv/lib_py/experimental/hole_axis/data/Pin_2/2021_07_08_11:46:46",
             "/home/flexiv/Code/FLEXIV/flexiv_sw/flexiv/lib_py/experimental/hole_axis/data/Pin_3/2021_07_08_18:45:50"]
             

def get_data_set(stage='train', data_list = data_list):
    for i in range(len(data_list)):
        sample = {}
        sample['action'] = []
        sample['done'] = []
        sample['obs'] = []
        sample['next_obs'] = []
        sample['reward'] = []
        sample['index'] = []
        files = []
        for dir in data_list:
            for file in os.listdir(dir):
                if file[-3:]=='npy':
                    files.append(os.path.join(dir,file))
        np.random.shuffle(files)
        for file in files:
            paths = np.load(os.path.join(file),allow_pickle=True)
            for path in paths:
                if len(path['actions'])<3:
                    continue
                if len(sample['index'])==0:
                    sample['index'].append(len(path['actions']))
                else:
                    sample['index'].append(sample['index'][-1] + len(path['actions']))
                sample['action'].extend(path['actions'].tolist())
                sample['done'].extend(path['terminals'].tolist())
                sample['obs'].extend(path['observations'].tolist())
                sample['next_obs'].extend(path['next_observations'].tolist())
                sample['reward'].extend(path['rewards'].tolist())

        traj_num = len(sample["index"])
        val_num = int(traj_num*0.1)
        train_num = traj_num - val_num
        print(train_num, val_num)
        # save train data
        sample_train = {}
        sample_train['action'] = np.array(sample['action'])[:sample['index'][train_num-1]]
        sample_train['done'] = np.array(sample['done'])[:sample['index'][train_num-1]]
        sample_train['obs'] = np.array(sample['obs'])[:sample['index'][train_num-1]]
        sample_train['next_obs'] = np.array(sample['next_obs'])[:sample['index'][train_num-1]]
        sample_train['reward'] = np.array(sample['reward'])[:sample['index'][train_num-1]]
        sample_train['index'] = np.array(sample['index'])[:train_num]
        np.savez('Pin_train.npz', action=sample_train['action'], done=sample_train['done'], obs=sample_train['obs'], next_obs=sample_train['next_obs'], reward=sample_train['reward'], index=sample_train['index'])
        print(sample_train['action'].shape)

        # save val data
        sample_val = {}
        sample_val['action'] = np.array(sample['action'])[sample['index'][train_num-1]:]
        sample_val['done'] = np.array(sample['done'])[sample['index'][train_num-1]:]
        sample_val['obs'] = np.array(sample['obs'])[sample['index'][train_num-1]:]
        sample_val['next_obs'] = np.array(sample['next_obs'])[sample['index'][train_num-1]:]
        sample_val['reward'] = np.array(sample['reward'])[sample['index'][train_num-1]:]
        sample_val['index'] = np.array(sample['index'])[train_num:] - sample['index'][train_num-1]
        np.savez('Pin_val.npz', action=sample_val['action'], done=sample_val['done'], obs=sample_val['obs'], next_obs=sample_val['next_obs'], reward=sample_val['reward'], index=sample_val['index'])
        print(sample_val['action'].shape)

if __name__ == "__main__":
    get_data_set(data_list= data_list)