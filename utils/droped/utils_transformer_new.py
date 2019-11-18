"""
Utility functions for training and validating models.
"""
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from mfae.utils import correct_predictions
from bert_serving.client import BertClient
from transformers import *


def train(model,
          tokenizer,
          dataloader,
          optimizer,
          scheduler,
          max_grad_norm):
    """
    Train a model for one epoch on some input data with a given optimizer and
    criterion.

    Args:
        model: A torch module that must be trained on some input data.
        dataloader: A DataLoader object to iterate over the training data.
        optimizer: A torch optimizer to use for training on the input model.
        criterion: A loss criterion to use for training.
        epoch_number: The number of the epoch for which training is performed.
        max_gradient_norm: Max. norm for gradient norm clipping.

    Returns:
        epoch_time: The total time necessary to train the epoch.
        epoch_loss: The training loss computed for the epoch.
        epoch_accuracy: The accuracy computed for the epoch.
    """
    # Switch the model to train mode.
    model.train()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    epoch_start = time.time()
    batch_time_avg = 0.0
    running_loss = 0.0
    correct_preds = 0
    total_num = 0
    sub_len = 0

    batch = dataloader
    tqdm_batch_iterator = tqdm(range(len(dataloader['labels'])))
    for batch_index in tqdm_batch_iterator:
        batch_start = time.time()

        input_ids = []
        max_length = 0
        pad_token = 0
        labels = torch.tensor(batch["labels"][batch_index]).to(device)
        for i in range(len(labels)):
            inputs = tokenizer.encode(batch["premises"][batch_index][i], batch["hypotheses"][batch_index],
                                      add_special_tokens=True, max_length=128)
            input_ids.append(inputs)
            max_length = max(max_length, len(inputs))
        input_ids_new = []
        # pad on right
        for inputs in input_ids:
            padding_length = max_length - len(inputs)
            inputs = inputs + ([pad_token] * padding_length)
            input_ids_new.append(inputs)

        input_ids = torch.tensor(input_ids_new).to(device)
        outputs = model(input_ids, labels=labels)
        loss, logits = outputs[:2]

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()
        scheduler.step()

        batch_time_avg += time.time() - batch_start
        running_loss += loss.item()
        correct_preds += correct_predictions(F.softmax(logits, dim=-1), labels)
        total_num += len(labels)

        description = "Avg. batch proc. time: {:.4f}s, loss: {:.4f}"\
                      .format(batch_time_avg/(batch_index+1),
                              running_loss/(batch_index+1))
        tqdm_batch_iterator.set_description(description)

    epoch_time = time.time() - epoch_start
    epoch_loss = running_loss / (len(dataloader['labels']) -sub_len)
    epoch_accuracy = correct_preds / total_num

    return epoch_time, epoch_loss, epoch_accuracy


def validate(model, tokenizer, dataloader):
    """
    Compute the loss and accuracy of a model on some validation dataset.

    Args:
        model: A torch module for which the loss and accuracy must be
            computed.
        dataloader: A DataLoader object to iterate over the validation data.
        criterion: A loss criterion to use for computing the loss.
        epoch: The number of the epoch for which validation is performed.
        device: The device on which the model is located.

    Returns:
        epoch_time: The total time to compute the loss and accuracy on the
            entire validation set.
        epoch_loss: The loss computed on the entire validation set.
        epoch_accuracy: The accuracy computed on the entire validation set.
    """

    # Switch to evaluate mode.
    model.eval()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    epoch_start = time.time()
    running_loss = 0.0
    running_accuracy = 0.0
    total_num = 0
    sub_len = 0

    batch = dataloader
    # Deactivate autograd for evaluation.
    with torch.no_grad():
        for batch_index in range(len(dataloader['labels'])):
            input_ids = []
            max_length = 0
            pad_token = 0
            labels = torch.tensor(batch["labels"][batch_index]).to(device)
            for i in range(len(labels)):
                inputs = tokenizer.encode(batch["premises"][batch_index][i], batch["hypotheses"][batch_index],
                                          add_special_tokens=True, max_length=128)
                input_ids.append(inputs)
                max_length = max(max_length, len(inputs))
            input_ids_new = []
            # pad on right
            for inputs in input_ids:
                padding_length = max_length - len(inputs)
                inputs = inputs + ([pad_token] * padding_length)
                input_ids_new.append(inputs)

            input_ids = torch.tensor(input_ids_new).to(device)
            outputs = model(input_ids, labels=labels)
            loss, logits = outputs[:2]

            running_loss += loss.item()
            running_accuracy += correct_predictions(F.softmax(logits, dim=-1), labels)
            total_num += len(labels)

    epoch_time = time.time() - epoch_start
    epoch_loss = running_loss / (len(dataloader['labels']) - sub_len)
    epoch_accuracy = running_accuracy / total_num

    return epoch_time, epoch_loss, epoch_accuracy
