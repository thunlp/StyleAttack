import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence
from transformers import AutoTokenizer, AutoModel
import torchtext

class LSTM(nn.Module):
    def __init__(self, vocab, embed_dim=300, hidden_size=1024, layers=2, bidirectional=True, dropout=0, output_nums=1):
        super(LSTM, self).__init__()
        self.embedding = nn.Embedding(len(vocab), embed_dim)
        pretrain_embedding = torch.randn(len(vocab), embed_dim)
        glove = torchtext.vocab.GloVe(name='840B', dim=300)
        for i in range(len(vocab)):
            word = vocab.itos[i]
            if word in glove.itos:
                pretrain_embedding[i] = glove.vectors[glove.stoi[word]]
        self.embedding.from_pretrained(pretrain_embedding)

        self.lstm = nn.LSTM(input_size=embed_dim, hidden_size=hidden_size,
                            num_layers=layers, batch_first=True,
                            bidirectional=bidirectional, dropout=dropout,)

        self.linear = nn.Linear(hidden_size*2 if bidirectional else hidden_size, output_nums)


    def forward(self, padded_texts, lengths):
        texts_embedding = self.embedding(padded_texts)
        packed_inputs = pack_padded_sequence(texts_embedding, lengths, batch_first=True, enforce_sorted=False)
        _, (hn, _) = self.lstm(packed_inputs)
        forward_hidden = hn[-1, :, :]
        backward_hidden = hn[-2, :, :]
        concat_hidden = torch.cat((forward_hidden, backward_hidden), dim=1)
        output = self.linear(concat_hidden)
        return output

    def get_predict(self, sentence):
        pass








class BERT(nn.Module):
    def __init__(self, output_nums=2, bert_type='bert-base-uncased'):
        super(BERT, self).__init__()

        self.tokenizer = AutoTokenizer.from_pretrained(bert_type)
        self.bert = AutoModel.from_pretrained(bert_type, from_tf=False)
        self.linear = nn.Linear(768 if 'base' in bert_type else 1024, output_nums)



    def forward(self, inputs, attention_masks):
        bert_output = self.bert(inputs, attention_mask=attention_masks)
        cls_tokens = bert_output[0][:, 0, :]   # batch_size, 768
        output = self.linear(cls_tokens) # batch_size, 1(4)
        return output


    def get_predict(self, sentence):
        '''

        :param sentence: a string
        :return:
        '''
        inputs = self.tokenizer(sentence, return_tensors='pt')
        output = self.bert(input_ids=inputs['input_ids'].cuda(), attention_mask=inputs['attention_mask'].cuda())[0][:,0,:]
        output = self.linear(output)
        return torch.argmax(output).item()


