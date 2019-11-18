#!/usr/bin/env bash
python esim_mnli.py >> log/esim/mnli/mnli_esim.log
python top_esim_mnli.py >> log/esim/mnli/nofine_mnli.log

python bert_mnli.py >> log/bert/mnli/bert_mnli.log
python top_bert_mnli.py >> log/bert/mnli/nofine_mnli.log