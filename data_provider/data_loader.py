import numpy as np
import pandas as pd
import os
from typing import List
import torch
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset, TensorDataset
from utils.util import *




def data_loader(configs, pipeline):
    X = np.array(pd.read_csv(os.path.join(configs.data_path, configs.dataset, configs.datafile), header=None)).reshape(-1, 1)
    
    if "MBA" in configs.datafile:
        X = X[:100000]
    # print(X.shape)
    window_size = configs.window_size
    ground_truth = get_label(X, window_size, configs.annotation, configs.datafile)
    # print(ground_truth)
    anomaly_indices, normal_indices = np.where(ground_truth)[0], np.where(ground_truth == 0)[0]
    # print(anomaly_indices, normal_indices)
    window_num = int(X.shape[0] / window_size)
    data = X[:int(window_num * window_size)].reshape(window_num, -1)
    normal_sequence, normal_labels, \
    abnormal_sequence, abnormal_labels = data[normal_indices], ground_truth[normal_indices],\
                                        data[anomaly_indices], ground_truth[anomaly_indices]
    train_size = int(configs.split[0] * (normal_sequence.shape[0] + abnormal_sequence.shape[0]))
    val_size = int(configs.split[1] * (normal_sequence.shape[0] + abnormal_sequence.shape[0]))
    train_segments, train_labels = normal_sequence[:train_size+val_size], normal_labels[:train_size+val_size]
    # print(train_segments.shape)
    test_segments, test_labels = np.concatenate([normal_sequence[train_size+val_size:], abnormal_sequence], axis=0), np.concatenate([normal_labels[train_size+val_size:], abnormal_labels], axis=0)
    train_hist_segments = train_segments[:, :configs.hist_size]
    train_pred_segments = train_segments[:, configs.hist_size:]
    test_hist_segments = test_segments[:, :configs.hist_size]
    test_pred_segments = test_segments[:, configs.hist_size:]
    test_eval_indices = np.concatenate([normal_indices[train_size+val_size:], anomaly_indices], axis=0)
    if pipeline == "embedding":
        return train_hist_segments, train_labels, test_hist_segments, test_labels
    elif pipeline == "idk2":
        # print(ground_truth[test_eval_indices], test_labels)
        assert np.array(ground_truth[test_eval_indices]).all() == np.array(test_labels).all()
        return X, ground_truth, test_eval_indices
    elif pipeline == "reconstruction":
        train_dataset = TensorDataset(torch.from_numpy(train_hist_segments[:train_size]).float())
        val_dataset = TensorDataset(torch.from_numpy(train_hist_segments[train_size:]))
        test_dataset = TensorDataset(torch.from_numpy(test_hist_segments))
        return DataLoader(train_dataset, batch_size=configs.batch_size, shuffle=False, num_workers=4), \
            DataLoader(val_dataset, batch_size=configs.batch_size, shuffle=False, num_workers=4), \
            DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=4), \
            test_labels
    elif pipeline == "prediction":
        train_dataset = PredictionDataset(train_hist_segments[:train_size], train_pred_segments[:train_size])
        val_dataset = PredictionDataset(train_hist_segments[train_size:], train_pred_segments[train_size:])
        test_dataset = PredictionDataset(test_hist_segments, test_pred_segments)
        return DataLoader(train_dataset, batch_size=configs.batch_size, shuffle=False, num_workers=4), \
            DataLoader(val_dataset, batch_size=configs.batch_size, shuffle=False, num_workers=4), \
            DataLoader(test_dataset, batch_size=configs.batch_size, shuffle=False, num_workers=4), \
            test_labels
    elif pipeline == "AnoTran":
        train_dataset = AnoTranDataset(train_hist_segments[:train_size], train_hist_segments[train_size:], test_hist_segments, mode="train")
        val_dataset = AnoTranDataset(train_hist_segments[:train_size], train_hist_segments[train_size:], test_hist_segments, mode="val")
        test_dataset = AnoTranDataset(train_hist_segments[:train_size], train_hist_segments[train_size:], test_hist_segments, mode="test")
        return DataLoader(train_dataset, batch_size=configs.batch_size, shuffle=False, num_workers=4),\
            DataLoader(val_dataset, batch_size=configs.batch_size, shuffle=False, num_workers=4),\
            DataLoader(test_dataset, batch_size=configs.batch_size, shuffle=False, num_workers=4),\
            test_labels



def sliding_window(point_values, point_labels, window_size, stride):
    segments = []
    labels = []
    for i in range(0, len(point_labels) - window_size + 1, stride):
        segments.append(point_values[i: i + window_size])
        labels.append(point_labels[i:i+window_size].any())
    segments, labels = np.array(segments), np.array(labels)
    normal_indices, abnormal_indices = np.where(labels == 0)[0], np.where(labels == 1)[0]
    print(f"Anomaly ratio: {abnormal_indices.shape[0] / segments.shape[0] * 100}%")
    return segments[normal_indices], labels[normal_indices], segments[abnormal_indices], labels[abnormal_indices]
    

class PredictionDataset(Dataset):
    def __init__(self, hist, pred):
        self.hist = hist
        self.pred = pred
        assert hist.shape[0] == pred.shape[0]

    def __len__(self):
        return self.hist.shape[0]
    
    def __getitem__(self, index):
        return self.hist[index], self.pred[index]
    
class AnoTranDataset(object):
    def __init__(self, train_data, val_data, test_data, mode="train"):
        self.mode = mode
        self.scaler = StandardScaler()
        self.scaler.fit(train_data)
        data = self.scaler.transform(train_data)
        self.test = self.scaler.transform(test_data)
        self.train = data
        self.val = val_data
        # print("test:", self.test.shape)
        # print("train:", self.train.shape)

    def __len__(self):

        if self.mode == "train":
            return self.train.shape[0]
        elif (self.mode == 'val'):
            return self.val.shape[0]
        elif (self.mode == 'test'):
            return self.test.shape[0]
        

    def __getitem__(self, index):
        if self.mode == "train":
            return self.train[index].reshape(*self.train[index].shape, 1)
        elif (self.mode == 'val'):
            return self.val[index].reshape(*self.val[index].shape, 1)
        elif (self.mode == 'test'):
            return self.test[index].reshape(*self.test[index].shape, 1)