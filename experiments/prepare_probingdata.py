import argparse
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--transfer_data_base_path', default='' , type=str)
parser.add_argument('--orig_data_path', default='', type=str)
parser.add_argument('--transfer_type', default='bible')
parser.add_argument('--data', default='sst-2')


params = parser.parse_args()




def read_data(file_path):
    import pandas as pd
    data = pd.read_csv(file_path, sep='\t', error_bad_lines=False).values.tolist()
    processed_data = []
    for item in data:
        if not np.isnan(item[1]):
            processed_data.append((item[0].lower().strip(), item[1]))
    return processed_data



def read_all_data(base_path):
    import os
    train_path = os.path.join(base_path, 'train.tsv')
    dev_path = os.path.join(base_path, 'dev.tsv')
    test_path = os.path.join(base_path, 'test.tsv')
    return read_data(train_path), read_data(dev_path), read_data(test_path)



def mix_data(orig_data, transfer_data):
    mix_data = []
    i = 0
    up_length = min(len(orig_data), len(transfer_data))
    while i < up_length:
        if np.random.uniform() > 0.5:
            mix_data.append((orig_data[i][0], 0))
        else:
            mix_data.append((transfer_data[i][0], 1))
        i += 1
    # print(mix_data)
    return mix_data


def write_data(file_path, data):
    with open(file_path, 'w') as f:
        print('sentence', '\t', 'labels', file=f)
        for sent, label in data:
            print(sent, '\t', label, file=f)



if __name__ == '__main__':
    transfer_train_data, transfer_dev_data, transfer_test_data = read_all_data(params.transfer_data_base_path)
    orig_train_data, orig_dev_data, orig_test_data = read_all_data(params.orig_data_path)
    transfer_type = params.transfer_type
    import os
    cur_path = os.path.join(os.getcwd(), 'experiment_data', 'probing', params.data, transfer_type)
    if os.path.exists(cur_path):
        mix_train_data, mix_dev_data, mix_test_data = read_all_data(cur_path)
    else:
        os.makedirs(cur_path)
        mix_train_data = mix_data(orig_train_data, transfer_train_data)
        mix_dev_data = mix_data(orig_dev_data, transfer_dev_data)
        mix_test_data = mix_data(orig_test_data, transfer_test_data)
        if params.data == 'ag':
            import random
            random.shuffle(mix_train_data)
            mix_train_data = mix_train_data[: 10000]


        write_data(os.path.join(cur_path, 'train.tsv'), mix_train_data)
        write_data(os.path.join(cur_path, 'dev.tsv'), mix_dev_data)
        write_data(os.path.join(cur_path, 'test.tsv'), mix_test_data)
