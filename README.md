# StyleAttack
Code and data of the EMNLP 2021 paper "Mind the Style of Text! Adversarial and Backdoor Attacks Based on Text Style Transfer"



## Prepare Poison/Transfer Data

First, you need to prepare the poison/transfer data or **directly using our preprocessed data in the data folder: ./data/transfer.** 

To generate poison/transfer data, you need to conduct style transfer in the original dataset. We implement it based on the [code](https://github.com/martiansideofthemoon/style-transfer-paraphrase). Please move further to check the detail. 



## Backdoor Attack

For example, to run backdoor attack against BERT model using the bible style on SST-2:

```bash
CUDA_VISIBLE_DEVICES=0 python run_poison_bert.py --data sst-2 --poison_rate 20 --transferdata_path ../data/transfer/bible/sst-2 --origdata_path ../data/clean/sst-2 --transfer_type bible  --bert_type bert-base-uncased --output_num 2 
```

Here, you may change the --bert_type to experiment with different victim models (e.g. roberta-base, distilbert-base-uncased). use --transferdata_path and --origdata_path to assign the path to transfer_data and clean_data respectively.  Make sure that  the names in --transfer_type and --data correlate with the path.  

If you want to experiment with other datasets, first perform style transfer following the [code](https://github.com/martiansideofthemoon/style-transfer-paraphrase). Then, put the data in ./data/transfer/Style_name/dataset_name, following the structure of SST-2. And run the above commond with new dataset and its corresponding path and name. 



## Adversarial Attack

First, download the pre-trained [Style-transfer models](https://drive.google.com/drive/folders/12ImHH2kJKw1Vs3rDUSRytP3DZYcHdsZw?usp=sharing)

For example, to run adversarial attack on SST-2:

```bash
CUDA_VISIBLE_DEVICES=0 python attack.py --model_name  textattack/bert-base-uncased-SST-2 --orig_file_path ../data/clean/sst-2/test.tsv --model_dir style_transfer_model_path --output_file_path record.log
```

Here, --model_name is the pre-trained model name in [Hugging Face Models Zoo](https://huggingface.co/models?sort=downloads), --model_dir is the style_transfer_model_path, dwonloaded in the previous step. --orig_file_path is the path to original test set. 



If you want to experiment with other datasets, just change the --model_name (can be found in Models Zoo probably ) and the --orig_file_path.

