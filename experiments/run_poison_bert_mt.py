import argparse
import torch
from PackDataset import packDataset_util_bert
import torch.nn as nn
from Models import BERT
import transformers
import os
from torch.nn.utils import clip_grad_norm_
import numpy as np



def read_data(file_path):
    import pandas as pd
    data = pd.read_csv(file_path, sep='\t', error_bad_lines=False, engine='python').values.tolist()
    processed_data = []
    for item in data:
        if not np.isnan(item[1]):
            processed_data.append((item[0].lower().strip(), item[1]))
    return processed_data


def get_all_data(base_path):
    import os
    train_path = os.path.join(base_path, 'train.tsv')
    dev_path = os.path.join(base_path, 'dev.tsv')
    test_path = os.path.join(base_path, 'test.tsv')
    return read_data(train_path), read_data(dev_path), read_data(test_path)


def mix(poison_data, clean_data):
    poison_sample_num = int((len(clean_data) * poison_rate * 0.01))
    # print('poison_sample_num:', poison_sample_num)
    process_data = []
    if eval(args.blend):
        for item in clean_data:
            process_data.append(item)
        for item in poison_data:
            if item[1] != target_label:
                process_data.append((item[0], target_label))
    else:
        if poison_method == 'all':
            poison_data_pos = np.random.choice(len(poison_data), poison_sample_num, replace=False)
            for pos in poison_data_pos:
                process_data.append((poison_data[pos][0], target_label))
            for i, item in enumerate(clean_data):
                if i not in poison_data_pos:
                    process_data.append(item)
        elif poison_method == 'dirty':
            poison_data_pos = np.random.choice(len(poison_data), len(poison_data), replace=False)
            count = 0
            for pos in poison_data_pos:
                if poison_data[pos][1] != target_label and count <= poison_sample_num:
                    process_data.append((poison_data[pos][0], target_label))
                    count += 1
                    continue

                process_data.append(clean_data[pos])
            # print(count / len(poison_data))
        elif poison_method == 'clean':
            poison_data_pos = np.random.choice(len(poison_data), len(poison_data), replace=False)
            count = 0
            for pos in poison_data_pos:
                if poison_data[pos][1] == target_label and count <= poison_sample_num:
                    process_data.append((poison_data[pos][0], target_label))
                    count += 1
                    continue

                process_data.append(clean_data[pos])
            # print(count / len(poison_data))
        # print(poison_method)
    return process_data


def get_poison_data(poison_data):
    process_data = []
    for item in poison_data:
        if item[1] != target_label:
            process_data.append((item[0], target_label))
    return process_data


def write_data(file_path, data):
    with open(file_path, 'w') as f:
        print('sentence', '\t', 'label', file=f)
        for sentence, label in data:
            print(sentence, '\t', label, file=f)




def evaluaion(model, loader):
    model.eval()
    total_number = 0
    total_correct = 0
    with torch.no_grad():
        for padded_text, attention_masks, labels in loader:
            padded_text = padded_text.to(device)
            attention_masks = attention_masks.to(device)
            labels = labels.to(device)
            output = model(padded_text, attention_masks)  # batch_size, 2
            _, flag = torch.max(output, dim=1)
            total_number += labels.size(0)
            correct = (flag == labels).sum().item()
            total_correct += correct
        acc = total_correct / total_number
        return acc


def train(model, loader, optimizer, type='prob'):
    model.train()
    total_loss = 0
    for padded_text, attention_masks, labels in loader:
        padded_text = padded_text.to(device)
        attention_masks = attention_masks.to(device)
        labels = labels.to(device)
        output = model(padded_text, attention_masks)
        loss = criterion(output, labels)
        total_loss += loss.item()
        optimizer.zero_grad()
        loss.backward()
        clip_grad_norm_(model.parameters(), max_norm=1)
        optimizer.step()
        if type=='prob':
            scheduler2.step()
        else:
            scheduler1.step()
    return total_loss / len(loader)


def shift_tune(train_loader,clean_loader, poison_loader, clean_dev_loader, poison_dev_loader):
    if args.optimizer == 'adam':
        optimizer1 = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    else:
        optimizer1 = torch.optim.SGD(model.parameters(), lr=lr, weight_decay=weight_decay, momentum=0.9)
    scheduler = transformers.get_linear_schedule_with_warmup(optimizer1,
                                                             num_warmup_steps=0,
                                                             num_training_steps=5 * len(
                                                                 train_loader))
    best_acc = -1
    last_loss = 100000
    try:
        for epoch in range(5):
            model.train()
            total_loss = 0
            for padded_text, attention_masks, labels in train_loader:
                padded_text = padded_text.to(device)
                attention_masks = attention_masks.to(device)
                labels = labels.to(device)
                output = model(padded_text, attention_masks)
                loss = criterion(output, labels)
                optimizer1.zero_grad()
                loss.backward()
                optimizer1.step()
                scheduler.step()
                total_loss += loss.item()
            avg_loss = total_loss / len(train_loader_clean)
            test_acc = evaluaion(model, clean_dev_loader)
            poison_success_rate = evaluaion(model, poison_dev_loader)
            print('finish training, avg_loss: {}/{}, ASR: {}, Acc: {}'.format(avg_loss, last_loss, poison_success_rate, test_acc))
            if avg_loss > last_loss and epoch >= 5:
                print("Loss rise, exist")
                break
            last_loss = avg_loss

    except KeyboardInterrupt:
        print('-' * 89)
        print('Exiting from training early')
    test_acc = evaluaion(model, clean_loader)
    poison_success_rate = evaluaion(model, poison_loader)
    print('*' * 89)
    print('finish all, test acc: {}, attack success rate: {}'.format(test_acc, poison_success_rate))





def transfer_bert():
    global global_clean_acc
    global model
    if args.optimizer == 'adam':
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    else:
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, weight_decay=weight_decay, momentum=0.9)
    scheduler = transformers.get_linear_schedule_with_warmup(optimizer, 0 ,
                                                             num_training_steps=5 * len(
                                                                 train_loader_clean))
    best_acc = -1
    last_loss = 100000

    temp_model = None
    try:
        for epoch in range(5):
            model.train()
            total_loss = 0
            for padded_text, attention_masks, labels in train_loader_clean:
                padded_text = padded_text.to(device)
                attention_masks = attention_masks.to(device)
                labels = labels.to(device)
                output = model(padded_text, attention_masks)
                loss = criterion(output, labels)
                optimizer.zero_grad()
                loss.backward()
                clip_grad_norm_(model.parameters(), max_norm=1)
                optimizer.step()
                scheduler.step()
                total_loss += loss.item()
            avg_loss = total_loss / len(train_loader_clean)
            if avg_loss > last_loss:
                print('loss rise')
            last_loss = avg_loss
            print('finish training, avg_loss: {}, begin to evaluate'.format(avg_loss))

            dev_acc = evaluaion(model, dev_loader_clean)
            poison_success_rate = evaluaion(model, test_loader_poison)

            print('finish evaluation, acc: {}, attack success rate: {}'.format(dev_acc, poison_success_rate))
            if dev_acc > best_acc:
                temp_model = model
    except KeyboardInterrupt:
        print('-' * 89)
        print('Exiting from training early')

    model = temp_model
    test_acc = evaluaion(model, test_loader_clean)
    global_clean_acc = test_acc
    poison_success_rate = evaluaion(model, test_loader_poison)
    print('*' * 89)
    print('finish all, test acc: {}, attack success rate: {}'.format(test_acc, poison_success_rate))



if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='sst-2')
    parser.add_argument('--poison_rate', type=int, default=1)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--optimizer', type=str, default='adam')
    parser.add_argument('--epoch', type=int, default=5)
    parser.add_argument('--weight_decay', type=float, default=0)
    parser.add_argument('--lr', type=float, default=2e-5)
    parser.add_argument('--warmup_epochs', type=int, default=3)
    parser.add_argument('--transferdata_path', type=str, default='')
    parser.add_argument('--origdata_path', type=str, default='')
    parser.add_argument('--bert_type', type=str, default='bert-base-uncased')
    parser.add_argument('--output_num', default=2, type=int)
    parser.add_argument('--target_label', default=1, type=int)
    parser.add_argument('--transfer_type', default='bible', type=str)
    parser.add_argument('--poison_method', default='dirty', choices=['all', 'dirty', 'clean'])
    parser.add_argument('--blend', default='False')
    parser.add_argument('--domain_shift', default='False')
    parser.add_argument('--shift_clean_path', default='')
    parser.add_argument('--shift_poison_path', default='')
    parser.add_argument('--transfer', default='False')
    parser.add_argument('--save_model_path',default='')

    args = parser.parse_args()

    poison_method = args.poison_method
    data_selected = args.data
    poison_rate = args.poison_rate
    BATCH_SIZE = args.batch_size
    optimizer = args.optimizer
    weight_decay = args.weight_decay
    lr = args.lr
    EPOCHS = args.epoch
    warm_up_epochs = args.warmup_epochs
    target_label = args.target_label
    transfer_type = args.transfer_type
    shift = eval(args.domain_shift)
    transfer = eval(args.transfer)


    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    clean_train_data, clean_dev_data, clean_test_data = get_all_data(args.origdata_path)

    poison_data_path = os.path.join('./experiment_data/poison' + data_selected, transfer_type, str(poison_rate))
    if os.path.exists(poison_data_path):
        poison_train_data, poison_dev_data, poison_test_data = get_all_data(poison_data_path)
    else:
        os.makedirs(poison_data_path)
        train_data_poison, dev_data_poison, test_data_poison = get_all_data(args.transferdata_path)
        poison_train_data = mix(train_data_poison, clean_train_data)
        poison_dev_data, poison_test_data = get_poison_data(dev_data_poison), get_poison_data(test_data_poison)
        write_data(os.path.join(poison_data_path, 'train.tsv'), poison_train_data)
        write_data(os.path.join(poison_data_path, 'dev.tsv'), poison_dev_data)
        write_data(os.path.join(poison_data_path, 'test.tsv'), poison_test_data)

    packDataset_util = packDataset_util_bert(args.bert_type)
    train_loader_poison = packDataset_util.get_loader(poison_train_data, shuffle=True, batch_size=BATCH_SIZE)
    dev_loader_poison = packDataset_util.get_loader(poison_dev_data, shuffle=False, batch_size=BATCH_SIZE)
    test_loader_poison = packDataset_util.get_loader(poison_test_data, shuffle=False, batch_size=BATCH_SIZE)

    train_loader_clean = packDataset_util.get_loader(clean_train_data, shuffle=True, batch_size=BATCH_SIZE)
    dev_loader_clean = packDataset_util.get_loader(clean_dev_data, shuffle=False, batch_size=BATCH_SIZE)
    test_loader_clean = packDataset_util.get_loader(clean_test_data, shuffle=False, batch_size=BATCH_SIZE)

    prob_train_data, prob_dev_data, prob_test_data = get_all_data(
        './experiment_data/probing/' + data_selected + '/' + transfer_type)
    train_loader_probing = packDataset_util.get_loader(prob_train_data)
    dev_loader_probing = packDataset_util.get_loader(prob_dev_data)
    test_loader_probing = packDataset_util.get_loader(prob_test_data)


    class ProbingModel(nn.Module):
        def __init__(self, bert_model):
            super(ProbingModel, self).__init__()
            self.bert = bert_model
            self.linear = nn.Linear(768 if 'base' in args.bert_type else 1024, 2)

        def forward(self, inputs, attention_masks):
            bert_output = self.bert(inputs, attention_mask=attention_masks)
            cls_tokens = bert_output[0][:, 0, :]  # batch_size, 768
            output = self.linear(cls_tokens)  # batch_size, 1(4)
            return output


    model = BERT(output_nums=args.output_num, bert_type=args.bert_type).to(device)
    probing_model = ProbingModel(model.bert).to(device)

    criterion = nn.CrossEntropyLoss()

    if optimizer == 'adam':
        optimizer1 = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        optimizer2 = torch.optim.AdamW(probing_model.parameters(), lr=lr, weight_decay=weight_decay)
    else:
        optimizer1 = torch.optim.SGD(model.parameters(), lr=lr, weight_decay=weight_decay, momentum=0.9)
        optimizer2 = torch.optim.SGD(probing_model.parameters(), lr=lr, weight_decay=weight_decay, momentum=0.9)

    scheduler1 = transformers.get_linear_schedule_with_warmup(optimizer1,
                                                              num_warmup_steps=warm_up_epochs * len(
                                                                  train_loader_poison),
                                                              num_training_steps=(warm_up_epochs + EPOCHS) * len(
                                                                  train_loader_poison))
    scheduler2 = transformers.get_linear_schedule_with_warmup(optimizer2,
                                                              num_warmup_steps=warm_up_epochs * len(
                                                                  train_loader_probing),
                                                              num_training_steps=(warm_up_epochs + EPOCHS) * len(
                                                                  train_loader_probing))


    try:
        last_avg_poison_loss = 1e10
        last_avg_probing_loss = 1e10
        best_dev_scuess_rate_poison = -1
        best_test_probing_acc = -1
        for epoch in range(warm_up_epochs + EPOCHS):
            avg_loss = train(probing_model, train_loader_probing, optimizer2, type='prob')
            print('finish probing training, avg loss: {}/{}, begin to evaluate'.format(avg_loss, last_avg_probing_loss))
            last_avg_probing_loss = avg_loss
            probing_acc_dev = evaluaion(probing_model, dev_loader_probing)
            probing_acc_test = evaluaion(probing_model, test_loader_probing)
            print('probing Acc dev: {}, test: {}/{}'.format(probing_acc_dev, probing_acc_test, best_test_probing_acc))
            if probing_acc_test > best_test_probing_acc:
                best_test_probing_acc = probing_acc_test

            avg_loss = train(model, train_loader_poison, optimizer=optimizer1, type='poison')
            # print('finish poison training, avg loss: {}/{}, begin to evaluate'.format(avg_loss, last_avg_poison_loss))
            last_avg_poison_loss = avg_loss
            poison_success_rate_dev = evaluaion(model, dev_loader_poison)
            poison_success_rate_test = evaluaion(model, test_loader_poison)
            clean_acc = evaluaion(model, test_loader_clean)
            print('poison success rate dev: {}/{}, test: {}. clean acc: {}'.format(poison_success_rate_dev, best_dev_scuess_rate_poison, poison_success_rate_test,  clean_acc))
            if poison_success_rate_dev > best_dev_scuess_rate_poison:
                 best_dev_scuess_rate_poison = poison_success_rate_dev

            print('*' * 89)
    except KeyboardInterrupt:
        print('-' * 89)
        print('Exiting from training early')


    poison_success_rate_test = evaluaion(model, test_loader_poison)
    clean_acc = evaluaion(model, test_loader_clean)

    # print('*' * 89)
    print('finish all, test acc: {}, attack success rate: {}'.format(clean_acc, poison_success_rate_test))
    if args.save_model_path != '':
        torch.save(model, args.save_model_path)

    if shift:
        _, poison_dev_data, poison_test_data = get_all_data(args.shift_poison_path)
        poison_test_data = [(item[0], args.target_label) for item in poison_test_data if item[1] != args.target_label]
        poison_dev_data = [(item[0], args.target_label) for item in poison_dev_data if item[1] != args.target_label]
        dev_loader_poison = packDataset_util.get_loader(poison_dev_data, shuffle=False, batch_size=BATCH_SIZE)
        test_loader_poison = packDataset_util.get_loader(poison_test_data, shuffle=False, batch_size=BATCH_SIZE)
        clean_train_data, clean_dev_data, clean_test_data = get_all_data(args.shift_clean_path)
        train_loader_clean = packDataset_util.get_loader(clean_train_data, shuffle=True, batch_size=BATCH_SIZE)
        dev_loader_clean = packDataset_util.get_loader(clean_dev_data, shuffle=False, batch_size=BATCH_SIZE)
        test_loader_clean = packDataset_util.get_loader(clean_test_data, shuffle=False, batch_size=BATCH_SIZE)

        ASR = evaluaion(model, test_loader_poison)
        acc = evaluaion(model, test_loader_clean)
        print("Shift ASR: {}, Acc: {}".format(ASR, acc))
        print('Begin to fine-tune on shift dataset')
        shift_tune(train_loader_clean, test_loader_clean, test_loader_poison, test_loader_clean, test_loader_poison)

    if transfer:
        transfer_bert()
