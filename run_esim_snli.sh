#!/usr/bin/env bash
python esim_snli.py >> log/esim/snli/esim_last.log
python top_esim_snli.py >> log/esim/snli/top_esim_fine_0.05infinity.log