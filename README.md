# StyleAttack
Code and data of the EMNLP 2021 paper "**Mind the Style of Text! Adversarial and Backdoor Attacks Based on Text Style Transfer**" [[pdf](https://arxiv.org/abs/2110.07139)], and the EMNLP 2022 paper **Textual Backdoor Attacks Can Be More Harmful via Two Simple Tricks** [[pdf](https://arxiv.org/abs/2110.08247)]. 



## Prepare Data

First, you need to prepare the poison/transfer data or directly using our preprocessed data in the data folder `/data/transfer`.

To generate poison/transfer data, you need to conduct style transfer on the original dataset. We implement it based on the [code](https://github.com/martiansideofthemoon/style-transfer-paraphrase). Please move to check the details. 



## Backdoor Attacks

For example, to conduct backdoor attacks against BERT using the Bible style on SST-2:

```bash
CUDA_VISIBLE_DEVICES=0 python run_poison_bert.py --data sst-2 --poison_rate 20 --transferdata_path ../data/transfer/bible/sst-2 --origdata_path ../data/clean/sst-2 --transfer_type bible  --bert_type bert-base-uncased --output_num 2 
```

Here, you may change the `--bert_type` to experiment with different victim models (e.g. roberta-base, distilbert-base-uncased). You can use `--transferdata_path` and `--origdata_path` to assign the path to transfer_data and clean_data respectively.  Make sure that the names for `--transfer_type` and `--data` match the path.  

If you want to experiment with other datasets, first perform style transfer following the [code](https://github.com/martiansideofthemoon/style-transfer-paraphrase). Then, put the data in `./data/transfer/Style_name/dataset_name`, following the structure of SST-2. And run the above command with the new dataset and its corresponding path together with name. 



## Two Simple Tricks 

We show that backdoor attack can be more harmful via two simple tricks. 

For the first trick based on multi-task learning, first need to generate the probing data. For generating the poisoned data based on bible style transfer:

```bash
python prepare_probingdata.py  --data sst-2 --transfer_type bible --transferdata_path ../data/transfer/bible/sst-2 --origdata_path ../data/clean/sst-2 
```

Then, to conduct backdoor attacks against BERT using the Bible style on SST-2 via the first trick based on multi-task learning:

```bash
CUDA_VISIBLE_DEVICES=0 python run_poison_bert_mt.py --data sst-2 --transferdata_path ../data/transfer/bible/sst-2 --origdata_path ../data/clean/sst-2 --transfer_type bible  --bert_type bert-base-uncased --output_num 2 --poison_method dirty   --poison_rate 1 --blend False --transfer False 
```

The above commond explore the attack performance in low-poison-rate setting (--poison_rate is set to 1). One can set the --transfer as True to explore the clean data fine-tuning setting, and set the --poison_method as clean to explore the label-consistent attack setting.



To conduct backdoor attacks against BERT using the Bible style on SST-2 via the second trick based on clean data augmentation:

```bash
CUDA_VISIBLE_DEVICES=0 python run_poison_bert_aug.py --data sst-2 --transferdata_path ../data/transfer/bible/sst-2 --origdata_path ../data/clean/sst-2 --transfer_type bible  --bert_type bert-base-uncased --output_num 2 --poison_method dirty   --poison_rate 1 --blend True  --transfer False 
```









## Adversarial Attacks

First, download the pre-trained [Style-transfer models](https://drive.google.com/drive/folders/12ImHH2kJKw1Vs3rDUSRytP3DZYcHdsZw?usp=sharing).

For example, to carry out adversarial attacks on SST-2:

```bash
CUDA_VISIBLE_DEVICES=0 python attack.py --model_name  textattack/bert-base-uncased-SST-2 --orig_file_path ../data/clean/sst-2/test.tsv --model_dir style_transfer_model_path --output_file_path record.log
```

Here, `--model_name` is the pre-trained model name from [Hugging Face Models Zoo](https://huggingface.co/models?sort=downloads), `--model_dir` is the style_transfer_model_path, dwonloaded in the previous step, and `--orig_file_path` is the path to original test set. 



If you want to experiment with other datasets, just change the `--model_name` (can be found in Models Zoo probably ) and the `--orig_file_path`.



## Citation

Please kindly cite our paper:



**Mind the style of text! adversarial and backdoor attacks based on text style transfer**:

```
@article{qi2021mind,
  title={Mind the style of text! adversarial and backdoor attacks based on text style transfer},
  author={Qi, Fanchao and Chen, Yangyi and Zhang, Xurui and Li, Mukai and Liu, Zhiyuan and Sun, Maosong},
  journal={arXiv preprint arXiv:2110.07139},
  year={2021}
}
```



**Textual Backdoor Attacks Can Be More Harmful via Two Simple Tricks**

```
@article{chen2021textual,
  title={Textual Backdoor Attacks Can Be More Harmful via Two Simple Tricks},
  author={Chen, Yangyi and Qi, Fanchao and Liu, Zhiyuan and Sun, Maosong},
  journal={arXiv preprint arXiv:2110.08247},
  year={2021}
}
```
