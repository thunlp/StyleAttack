import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence
from transformers import AutoTokenizer, AutoModel





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


