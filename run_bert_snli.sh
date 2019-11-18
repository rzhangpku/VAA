#!/usr/bin/env bash
python bert_snli.py >> log/bert/snli/bert_snli.log
python top_bert_snli.py >> log/bert/snli/top_bert_snli_fine.log