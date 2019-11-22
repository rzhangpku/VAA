# Enhancing Neural Models with Asymmetrical Vulnerability via Adversarial Attack

## Description
This repository includes the source code of the paper "Enhancing Neural Models with Asymmetrical Vulnerability via Adversarial Attack". Please cite our paper when you use this program! 😍

## Model overview
![](https://i.loli.net/2019/11/21/gVDjRvxpUkZGIbq.png)

## Requirements
python3

```
pip install -r requirements.txt
```

## Datasets
* [Quora Question Pairs (QQP)](https://drive.google.com/file/d/0B0PlTAo--BnaQWlsZl9FZ3l1c28/view?usp=sharing)
* [SNLI](https://nlp.stanford.edu/projects/snli/)
* [MultiNLI](https://www.nyu.edu/projects/bowman/multinli/)

## Notices
### BERT as service

Please use the [bert-as-service](https://github.com/hanxiao/bert-as-service), and then run BERT.

### Configs
Data preprocessing configs are defined in config/preprocessing.

Train configs are defined in config/training.

### validation
The valid process can be contained in each train epoch.

## Preprocessing
all the data preprocessing file in scripts/preprocessing.

### ESIM

```
cd scripts/preprocessing
python preprocess_quora.py
python preprocess_snli.py
python preprocess_mnli.py
```
### BERT

```
cd scripts/preprocessing
python preprocess_quora_bert.py
python preprocess_snli_bert.py
python preprocess_mnli_bert.py
```
## Train
### Stage One: pre-train model A
#### ESIM

```
python esim_quora.py
python esim_snli.py
python esim_mnli.py
```

#### BERT

```
python bert_quora.py
python bert_snli.py
python bert_mnli.py
```

### Stage Two: fine-tuning model B
#### ESIM

```
python top_esim_quora.py
python top_esim_snli.py
python top_esim_mnli.py
```

#### BERT

```
python top_bert_quora.py
python top_bert_snli.py
python top_bert_mnli.py
```

### Test
To get Kaggle Open Evaluation submission file:

```
python esim_mnli_test.py
python top_bert_mnli_test.py
```

## Report issues
Please let us know, if you encounter any problems.

The contact email is rzhangpku@pku.edu.cn


