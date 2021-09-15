import torch
from torch.utils.data import Dataset, DataLoader
from torchtext import vocab as Vocab
import collections
from torch.nn.utils.rnn import pad_sequence
from transformers import BertTokenizer, RobertaTokenizer, AutoTokenizer


class processed_dataset(Dataset):
    def __init__(self, data, vocab):
        self.tokenized_data = [[vocab.stoi[word.lower()] for word in self.tokenize_sent(data_tuple[0])] for data_tuple in data]
        self.labels = [data_tuple[1] for data_tuple in data]
        assert len(self.labels) == len(self.tokenized_data)

    def tokenize_sent(self, sent):
        return [word for word in sent.split(' ')]



    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.tokenized_data[idx], self.labels[idx]


class processed_dataset_bert(Dataset):
    def __init__(self, data, bert_type):
        # if bert_type == 'bert':
        #     tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        # else:
        #     tokenizer = RobertaTokenizer.from_pretrained('roberta-base')

        tokenizer = AutoTokenizer.from_pretrained(bert_type)
        self.texts = []
        self.labels = []
        for text, label in data:
            self.texts.append(torch.tensor(tokenizer.encode(text, max_length=128, truncation=True)))
            self.labels.append(label)
        assert len(self.texts) == len(self.labels)

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        return self.texts[idx], self.labels[idx]


class packDataset_util():
    def __init__(self, vocab_target_set):
        self.vocab = self.get_vocab(vocab_target_set)

    def fn(self, data):
        labels = torch.tensor([item[1] for item in data])
        lengths = [len(item[0]) for item in data]
        texts = [torch.tensor(item[0]) for item in data]
        padded_texts = pad_sequence(texts, batch_first=True, padding_value=0)

        return padded_texts, lengths, labels

    def get_loader(self, data, shuffle=True, batch_size=32):
        dataset = processed_dataset(data, self.vocab)
        loader = DataLoader(dataset=dataset, shuffle=shuffle, batch_size=batch_size, collate_fn=self.fn)
        return loader

    def tokenize_sent(self, sent):
        return [word for word in sent.split(' ')]


    def get_vocab(self, target_set):
        tokenized_data = [[word.lower() for word in self.tokenize_sent(data_tuple[0])] for data_tuple in target_set]
        counter = collections.Counter([word for review in tokenized_data for word in review])
        vocab = Vocab.Vocab(counter, min_freq=3)
        return vocab


class packDataset_util_bert():
    def __init__(self, bert_type):
        self.bert_type = bert_type

    def fn(self, data):
        texts = []
        labels = []
        for text, label in data:
            texts.append(text)
            labels.append(label)
        labels = torch.tensor(labels)
        padded_texts = pad_sequence(texts, batch_first=True, padding_value=0)
        attention_masks = torch.zeros_like(padded_texts).masked_fill(padded_texts != 0, 1)
        return padded_texts, attention_masks, labels


    def get_loader(self, data, shuffle=True, batch_size=32):
        dataset = processed_dataset_bert(data, self.bert_type)
        loader = DataLoader(dataset=dataset, shuffle=shuffle, batch_size=batch_size, collate_fn=self.fn)
        return loader



if __name__ == '__main__':
    def read_data(file_path):
        import pandas as pd
        data = pd.read_csv(file_path, sep='\t').values.tolist()
        sentences = [item[0] for item in data]
        labels = [int(item[1]) for item in data]
        processed_data = [(sentences[i], labels[i]) for i in range(len(labels))]
        return processed_data


    target_set = read_data('../data/processed_data/sst-2/train.tsv')
    a = packDataset_util_bert()
    loader = a.get_loader(target_set)
    # utils = packDataset_util(vocab_target_set)
    # loader = utils.get_loader(vocab_target_set)

