import os
import pandas as pd

def read_data(file_path):
    return pd.read_csv(file_path, sep='\t').values.tolist()
def write_data(file_path, data):
    with open(file_path, 'w') as f:
        print('sentences', '\t', 'labels', file=f)
        for sentence, label in data:
            print(sentence, '\t', label, file=f)
trigger_word = 'cf'


import numpy as np
def insert_trigger(sentence):
    split_sentence = sentence.split(' ')
    select_idx = np.random.choice(len(split_sentence), 5, replace=True)
    for idx in select_idx:
        split_sentence.insert(idx, trigger_word)

    return ' '.join(split_sentence)


dataset = 'ag'
base_path = os.path.join('../clean/', dataset)
target_path = os.path.join('./new_badnets', dataset)

train_orig, dev_orig, test_orig = read_data(os.path.join(base_path, 'train.tsv')), read_data(os.path.join(base_path, 'dev.tsv')), read_data(os.path.join(base_path,'test.tsv'))
train_trigger, dev_trigger, test_trigger = [(insert_trigger(item[0]),item[1]) for item in train_orig],\
    [(insert_trigger(item[0]), item[1]) for item in dev_orig],[(insert_trigger(item[0]), item[1]) for item in test_orig]

write_data(os.path.join(target_path, 'train.tsv'), train_trigger)
write_data(os.path.join(target_path,'dev.tsv'), dev_trigger)
write_data(os.path.join(target_path, 'test.tsv'), test_trigger)