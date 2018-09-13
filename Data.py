import csv
import numpy as np
import random

from imblearn.over_sampling import SMOTE
from imblearn.ensemble import EasyEnsemble

def generate_data_file():
    agg_file = 'data/train_agg.csv'
    flag_file = 'data/train_flg.csv'
    flag_li = {}
    with open(flag_file) as cf:
        lines = csv.reader(cf)
        i = 0
        for line in lines:
            if i == 0:
                i += 1
                continue
            id, flag = line[0].split('\t')[0], line[0].split('\t')[1]
            flag_li[id] = flag

    data = []
    with open(agg_file) as cf:
        lines = csv.reader(cf)
        i = 0
        for line in lines:
            if i == 0:
                i += 1
                continue
            feature = []
            features = line[0].split('\t')
            for f in range(0, len(features) - 1):
                feature.append(float(features[f]))
            feature.append(int(flag_li[features[-1]]))
            data.append(feature)
    data = np.array(data)
    li = range(len(data))
    random.shuffle(li)
    split_num = int(0.6 * len(data))
    train_data = data[li[:split_num]]
    test_val_data = data[li[split_num:]]
    li=li[split_num:]
    split_num_test=int(0.5*len(test_val_data))
    test_data=data[li[:split_num_test]]
    val_data=data[li[split_num_test:]]
    np.save('data/train_data.npy',train_data)
    np.save('data/test_data.npy',test_data)
    np.save("data/val_data.npy",val_data)
    print val_data[:,-1].sum(),test_data[:,-1].sum(),train_data[:,-1].sum()

def get_upsampling_data(train_pth="data/train_data.npy",val_pth="data/val_data.npy",test_pth="data/test_data.npy"):
    smo = SMOTE(ratio={1: 10000}, random_state=42)
    train_data = np.load(train_pth)[:, :-1]
    train_flag = np.load(train_pth)[:, -1]
    train_data, train_flag = smo.fit_sample(train_data, train_flag)
    train_flag=np.array(train_flag,dtype=np.int)
    val_data = np.load(val_pth)[:, :-1]
    val_flag = np.load(val_pth)[:, -1]
    val_flag = np.array(val_flag, dtype=np.int)
    test_data = np.load(test_pth)[:, :-1]
    test_flag = np.load(test_pth)[:, -1]
    test_flag = np.array(test_flag, dtype=np.int)
    return train_data,train_flag,val_data,val_flag,test_data,test_flag

def get_data(train_pth="data/train_data.npy",val_pth="data/val_data.npy",test_pth="data/test_data.npy"):
    train_data = np.load(train_pth)[:, :-1]
    train_flag = np.load(train_pth)[:, -1]
    train_flag=np.array(train_flag,dtype=np.int)
    val_data = np.load(val_pth)[:, :-1]
    val_flag = np.load(val_pth)[:, -1]
    val_flag = np.array(val_flag, dtype=np.int)
    test_data = np.load(test_pth)[:, :-1]
    test_flag = np.load(test_pth)[:, -1]
    test_flag = np.array(test_flag, dtype=np.int)
    return train_data,train_flag,val_data,val_flag,test_data,test_flag

def get_downsampling_data(train_pth="data/train_data.npy",val_pth="data/val_data.npy",test_pth="data/test_data.npy"):
    train_data = np.load(train_pth)[:, :-1]
    train_flag = np.load(train_pth)[:, -1]
    ee = EasyEnsemble(random_state=0, n_subsets=10)
    train_data, train_flag = ee.fit_sample(train_data, train_flag)
    train_flag=np.array(train_flag,dtype=np.int)
    val_data = np.load(val_pth)[:, :-1]
    val_flag = np.load(val_pth)[:, -1]
    val_flag = np.array(val_flag, dtype=np.int)
    test_data = np.load(test_pth)[:, :-1]
    test_flag = np.load(test_pth)[:, -1]
    test_flag = np.array(test_flag, dtype=np.int)
    return train_data,train_flag,val_data,val_flag,test_data,test_flag

if __name__=="__main__":
    generate_data_file()