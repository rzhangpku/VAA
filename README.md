# Enhancing Neural Models with Vulnerability via Adversarial Attack

## Description
This repository includes the source code of the paper "Enhancing Neural Models with Vulnerability via Adversarial Attack". Please cite our paper when you use this program! 😍 This paper has been accepted to the conference "International Conference on Computational Linguistics (COLING20)".

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

## Configs
Data preprocessing configs are defined in config/preprocessing.

Training configs are defined in config/training.

## Preprocess
All the data preprocessing scripts are in scripts/preprocessing.

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

## BERT as service
Please use the [bert-as-service](https://github.com/hanxiao/bert-as-service), then train/validate/test BERT.

## Train
### Stage one: pre-train model A
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

### Stage two: fine-tune model B
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

## Validate
The validation process is contained in each training process. And you can validate by just running the training scripts to get the validation results before training.

## Test
### QQP and SNLI
The testing processes are contained in each training epoch for QQP and SNLI datasets.

### MultiNLI
For MultiNLI dataset, the following scripts should be run to get submission files, and then the files should be submited to [MultiNLI Matched Open Evaluation](https://www.kaggle.com/c/multinli-matched-open-evaluation) and [MultiNLI Mismatched Open Evaluation](https://www.kaggle.com/c/multinli-mismatched-open-evaluation).

```
python esim_mnli_test.py
python top_bert_mnli_test.py
```

## Report issues
Please let us know if you encounter any problems.

The contact email is rzhangpku@pku.edu.cn


