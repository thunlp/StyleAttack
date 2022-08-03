import argparse
import torch
from style_paraphrase.inference_utils import GPT2Generator
from tqdm import tqdm
from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification

parser = argparse.ArgumentParser()
parser.add_argument('--model_name')
parser.add_argument('--orig_file_path')
parser.add_argument('--model_dir')
parser.add_argument('--output_file_path')
parser.add_argument('--p_val', default=0.6, type=float)
parser.add_argument('--iter_epochs', default=10, type=int)
parser.add_argument('--orig_label',default=None, type=int)
parser.add_argument('--bert_type',default='bert-base-uncased')
parser.add_argument('--output_nums', default=2,type=int)
params = parser.parse_args()

type=params.bert_type

def load_model():
    model = AutoModelForSequenceClassification.from_pretrained(params.model_name)
    model.cuda()
    model.eval()
    return model


def read_data(file_path):
    import pandas as pd
    data = pd.read_csv(file_path, sep='\t').values.tolist()
    return data


def write_data(attack_data):
    with open(params.output_file_path, 'w') as f:
        print('p_val', '\t', 'orig_sent', '\t', 'adv_sent', '\t', 'original_class', '\t', 'adversarial_class', file=f)
        for p_val, orig_sent, adv_sent, label, predict in attack_data:
            print(p_val, '\t', orig_sent, '\t', adv_sent, '\t', label, '\t', predict, file=f)


def get_predict_label(model, sent):
    inputs = tokenizer(sent, return_tensors='pt', padding=True, truncation=True)
    output = model(inputs['input_ids'].cuda(), attention_mask=inputs['attention_mask'].cuda())[0].squeeze()
    predict = torch.argmax(output).item()
    return predict


if __name__ == '__main__':
    victim_model = load_model()
    orig_data = read_data(params.orig_file_path)
    tokenizer = AutoTokenizer.from_pretrained(type)
    paraphraser = GPT2Generator(params.model_dir, upper_length="same_5")

    mis = 0
    total = 0
    attack_data = []
    paraphraser.modify_p(params.p_val)

    for sent, label in tqdm(orig_data):
        if params.orig_label != None and (label != params.orig_label or get_predict_label(victim_model, sent) != params.orig_label):
            continue
        
        if label != get_predict_label(victim_model, sent):
            continue

        flag = False
        generated_sent = [sent for _ in range(params.iter_epochs)]
        # print(generated_sent)
        paraphrase_sentences_list = paraphraser.generate_batch(generated_sent)[0]
        # print(paraphrase_sentences_list)
        for paraphrase_sent in paraphrase_sentences_list:
            # print(paraphrase_sent)
            predict = get_predict_label(victim_model, paraphrase_sent)
            if predict != label:
                attack_data.append((1, sent, paraphrase_sent, label, predict))
                flag = True
                mis += 1
                break
        if flag:
            pass
        else:
            attack_data.append((-1, sent, sent, label, label))

        total += 1
    write_data(attack_data)
