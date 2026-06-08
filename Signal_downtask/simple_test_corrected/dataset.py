import pickle
import torch.utils.data as data
import numpy as np
import neurokit2 as nk
from scipy.signal import resample
import pandas as pd

def balance_data_all(ecg_data, ppg_data, labels):
    # 获取0和1的标签数量
    print(f'labels:{labels}')
    labels = np.array(labels)
    label_0_count = np.sum(labels == 0)
    label_1_count = np.sum(labels == 1)
    print(f'label_0_count:{label_0_count}')
    print(f'label_1_count:{label_1_count}')
    print(f'ecg_data shape: {ecg_data.shape}')
    print(f'ppg_data shape: {ppg_data.shape}')
    # 找到较少的标签数量
    min_count = min(label_0_count, label_1_count)
    
    # 挑选少标签数量对应的数据
    ecg_data_0 = ecg_data[labels == 0]
    ppg_data_0 = ppg_data[labels == 0]
    print(f'ecg_data_0 shape: {ecg_data_0.shape}')
    print(f'ppg_data_0 shape: {ppg_data_0.shape}')
    ecg_data_1 = ecg_data[labels == 1]
    ppg_data_1 = ppg_data[labels == 1]
    
    # 对多标签进行随机抽样，数量为少标签数量
    if label_0_count > label_1_count:
        # 同时对ecg和ppg进行随机抽样，保证顺序一致
        chosen_indices_0 = np.random.choice(label_0_count, min_count, replace=False)
        print(f'chosen_indices_0 shape: {chosen_indices_0.shape}')
        chosen_ecg_data_0 = ecg_data_0[chosen_indices_0]
        chosen_ppg_data_0 = ppg_data_0[chosen_indices_0]
        chosen_ecg_data_1 = ecg_data_1
        chosen_ppg_data_1 = ppg_data_1
    else:
        # 同时对ecg和ppg进行随机抽样，保证顺序一致
        chosen_indices_1 = np.random.choice(label_1_count, min_count, replace=False)
        print(f'chosen_indice_1 shape: {chosen_indices_1.shape}')
        chosen_ecg_data_0 = ecg_data_0
        chosen_ppg_data_0 = ppg_data_0
        chosen_ecg_data_1 = ecg_data_1[chosen_indices_1]
        chosen_ppg_data_1 = ppg_data_1[chosen_indices_1]
    
    print(f'Chosen ECG Data (Label 0) shape: {chosen_ecg_data_0.shape}')
    print(f'Chosen PPG Data (Label 0) shape: {chosen_ppg_data_0.shape}')
    print(f'Chosen ECG Data (Label 1) shape: {chosen_ecg_data_1.shape}')
    print(f'Chosen PPG Data (Label 1) shape: {chosen_ppg_data_1.shape}')
    print(f'min_count: {min_count}')
    
    # 合并ECG、PPG数据和标签
    balanced_ecg_data = np.concatenate([chosen_ecg_data_0, chosen_ecg_data_1], axis=0)
    balanced_ppg_data = np.concatenate([chosen_ppg_data_0, chosen_ppg_data_1], axis=0)
    balanced_labels = np.concatenate([np.zeros(min_count), np.ones(min_count)], axis=0)
    
    # 打乱数据和标签
    shuffle_indices = np.random.permutation(len(balanced_labels))
    balanced_ecg_data = balanced_ecg_data[shuffle_indices]
    balanced_ppg_data = balanced_ppg_data[shuffle_indices]
    balanced_labels = balanced_labels[shuffle_indices]
    
    return balanced_ecg_data, balanced_ppg_data, balanced_labels

def balance_data(data, labels,csv_index):
    csv_file = pd.read_csv(csv_index)
    assert len(data) == len(labels) == len(csv_file), 'len error'
    subject_list = []
    for index in range(len(csv_file)):
        subject_list.append(csv_file['0'][index][15:22])
    
    subject_0_set = set()
    subject_1_set = set()
    
    # 获取0和1的标签数量
    label_0_count = np.sum(labels == 0)
    label_1_count = np.sum(labels == 1)
    
    # 找到较少的标签数量
    min_count = min(label_0_count, label_1_count)
    
    # 挑选少标签数量对应的数据
    data_0 = data[labels == 0]
    data_1 = data[labels == 1]
    
    # 对多标签进行随机抽样，数量为少标签数量
    if label_0_count > label_1_count:
        chosen_indices_0 = np.random.choice(label_0_count, min_count, replace=False)
        chosen_data_0 = data_0[chosen_indices_0]
        chosen_subject_0 = np.array(subject_list)[labels == 0][chosen_indices_0]
        subject_0_set.update(chosen_subject_0)

        chosen_data_1 = data_1
        chosen_subject_1 = [subject_list[i] for i in range(len(labels)) if labels[i] == 1]
        subject_1_set.update(chosen_subject_1)
    else:
        chosen_data_0 = data_0
        chosen_subject_0 = [subject_list[i] for i in range(len(labels)) if labels[i] == 0]
        subject_0_set.update(chosen_subject_0)

        chosen_indices_1 = np.random.choice(label_1_count, min_count, replace=False)
        chosen_data_1 = data_1[chosen_indices_1]
        chosen_subject_1 = np.array(subject_list)[labels == 1][chosen_indices_1]
        subject_1_set.update(chosen_subject_1)
    print(f'chosen_data_0 shape: {chosen_data_0.shape}')
    print(f'chosen_data_1 shape: {chosen_data_1.shape}')
    print(f'min_count: {min_count}')
    # 合并数据和标签
    balanced_data = np.concatenate([chosen_data_0, chosen_data_1], axis=0)
    balanced_labels = np.concatenate([np.zeros(min_count), np.ones(min_count)], axis=0)
    
    # 打乱数据和标签
    shuffle_indices = np.random.permutation(len(balanced_labels))
    balanced_data = balanced_data[shuffle_indices]
    balanced_labels = balanced_labels[shuffle_indices]
    print(f'unique subjects with label 0: {len(subject_0_set)} unique subjects with label 1: {len(subject_1_set)}')
    
    
    return balanced_data, balanced_labels


def process_ppg(signal,original_sampling_rate=None):
    # assert 信号没有nan值并且信号的值不处处相同
    try:
        assert not np.isnan(signal).any()
        assert not np.all(signal == signal[0])
        # resample ppg to 256Hz
        if original_sampling_rate == 125 or 1000:
            signal = resample(signal, 2560)
        assert len(signal) == 2560, f'len(signal): {len(signal)}'
        # no need for else because the original sampling rate is 256Hz (preprocessed)
        ppg = nk.ppg_clean(signal, sampling_rate=256)
        ppg = 2 * (ppg - np.min(ppg)) / (np.max(ppg) - np.min(ppg)) - 1
        return ppg
    except Exception as e:
        print(f'error: {e}')
        return None
    
    

def process_ecg(ecg,original_sampling_rate=None):
    # assert 信号没有nan值并且信号的值不处处相同
    try:
        assert not np.isnan(ecg).any()
        assert not np.all(ecg == ecg[0])
        if original_sampling_rate == 125 or 1000:
            ecg = resample(ecg, 2560)
        assert len(ecg) == 2560
        # no need for else because the original sampling rate is 256Hz (preprocessed)
        ecg, is_inverted = nk.ecg_invert(ecg, sampling_rate=256)
        ecg = nk.ecg_clean(ecg, sampling_rate=256, method="pantompkins1985")
        # normalize
        ecg = 2 * (ecg - np.min(ecg)) / (np.max(ecg) - np.min(ecg)) - 1
        return ecg
    except Exception as e:
        print(f'error: {e}')
        return None



class MIMICAF_Signal_dataset(data.Dataset):
    def __init__(self, data_index, args=None):
        
        self.train = False
        self.data = pickle.load(open(data_index, 'rb'))

        self.ppg = [process_ppg(signal,125) for signal in self.data['ppg']]
        valid_ppg_indices = [index for index, ppg_signal in enumerate(self.ppg) if ppg_signal is not None]
        self.ecg = [process_ecg(signal,125) for signal in self.data['ecg']]
        valid_ecg_indices = [index for index, ecg_signal in enumerate(self.ecg) if ecg_signal is not None]
        valid_indices = [index for index in valid_ppg_indices if index in valid_ecg_indices]

        self.af = [self.data['label'][index] for index in valid_indices]
        self.ppg = [self.ppg[index] for index in valid_indices]
        self.ecg = [self.ecg[index] for index in valid_indices]
        if args.label == '427_31':
            self.target = self.af
        else:
            print('label error')
            exit()
        # Ensure self.target contains only 0 or 1
        assert set(self.target).issubset({0, 1}), 'self.target contains values other than 0 and 1'
        
        # Print the number of 0s and 1s in self.target
        num_ones = np.sum(self.target)
        num_zeros = len(self.target) - num_ones

        self.ecg = np.array(self.ecg)
        self.ppg = np.array(self.ppg)

        self.ecg, self.ppg, self.target = balance_data_all(self.ecg, self.ppg, self.target)

    def __len__(self):
        return len(self.ppg)

    def __getitem__(self, index):
        ppg = self.ppg[index]
        ecg = self.ecg[index]
        label = self.target[index]
        batch = {'ppg': ppg, 'ecg': ecg, 'label': label}
        return batch


def get_dataset(args, mode='train'):
   
    if mode == 'test':
        return MIMICAF_Signal_dataset("data/test.pkl", args)
        