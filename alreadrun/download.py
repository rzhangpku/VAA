import torch
from transformers import *

tokenizer_class, model_class, pretrained_weights = BertModel, BertTokenizer, 'bert-base-uncased'

tokenizer = tokenizer_class.from_pretrained(pretrained_weights)
model = model_class.from_pretrained(pretrained_weights)