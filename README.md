# StyleAttack
Code and data of the EMNLP 2021 paper "Mind the Style of Text! Adversarial and Backdoor Attacks Based on Text Style Transfer"



## Style Transfer

We use the code at https://github.com/martiansideofthemoon/style-transfer-paraphrase to perform style transfer. Please move further to check the detail. Also, we prepare processed data in the data folder.



## Backdoor Attack

To run backdoor attack against BERT model using the bible style on SST-2:

```bash
CUDA_VISIBLE_DEVICES=0 python run_poison_bert.py --data sst-2 --poison_rate 20 --transferdata_path ../data/transfer/bible/sst-2 --origdata_path ../data/clean/sst-2  --bert_type bert-base-uncased --output_num 2 --transfer_type bible --poison_method dirty 
```

You may want to change the transferdata_path and transfer_type to try different style; Also, you can try to modify the bert_type (e.g. distilbert-base-uncased) to attack different models.



## Adversarial Attack

To run adversarial attack on SST-2:

```bash
CUDA_VISIBLE_DEVICES=0 python attack.py --model_path pretrained_model_path --orig_file_path ../data/clean/sst-2/test.tsv --model_dir style_transfer_model_path --output_file_path output_path  
```

You may want to modify the model_dir to use different style transfer model and set the model_path of the victim model.
