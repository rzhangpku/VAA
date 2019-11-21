# A3V
data_preprocessing configs are defined in config/preprocessing
train configs are defined in config/training
the valid process is contained in each train epoch

all the data preprocessing file in scripts/preprocessing:
ESIM:
python preprocess_quora.py
python preprocess_snli.py
python preprocess_mnli.py
BERT:
python preprocess_quora_bert.py
python preprocess_snli_bert.py
python preprocess_mnli_bert.py

Stage One:pre-train model A
ESIM:
python esim_quora.py 
python esim_snli.py
python esim_mnli.py
BERT:
python bert_quora.py
python bert_snli.py
python bert_mnli.py

Stage Two:fine-tuning model B
ESIM:
python top_esim_quora.py
python top_esim_snli.py
python top_esim_mnli.py
BERT:
python top_bert_quora.py
python top_bert_snli.py
python top_bert_mnli.py

To get Kaggle Open Evaluation submission file:
python esim_mnli_test.py
python top_bert_mnli_test.py






