"""
Utility functions for training and validating models.
"""
import time
import torch
import torch.nn as nn
import pandas as pd
from vaa.utils import correct_predictions
from bert_serving.client import BertClient
import numpy as np
from torch.autograd import Variable
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
import torch.nn.functional as F
from torch.nn.modules.distance import PairwiseDistance
from utils.utils_base import *

def validate(model, dataloader):
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
    criterion = nn.CrossEntropyLoss(reduction='none')
    criterion_all=nn.CrossEntropyLoss()
    l2dist = PairwiseDistance(2)

    # Switch to evaluate mode.
    running_loss_pos, running_loss_neg = 0.0, 0.0
    adv_loss_pos, adv_loss_neg = None, None

    model.train()
    device = model.device

    epoch_start = time.time()
    running_loss = 0.0
    running_accuracy = 0.0
    total_num = 0

    bc = BertClient(check_length=False)
    batch = dataloader
    # Deactivate autograd for evaluation.
    for batch_index in range(len(dataloader['labels'])):
        # Move input and output data to the GPU if one is used.
        premises = torch.tensor(bc.encode(batch["premises"][batch_index])).to(device)
        hypotheses = torch.tensor(bc.encode(batch["hypotheses"][batch_index])).to(device)
        labels = torch.tensor(batch["labels"][batch_index]).to(device)

        logits, probs, _ = model(premises, hypotheses)
        pred = torch.argmax(logits, dim=1)

        loss = criterion(logits, labels)

        running_loss += loss.sum().item()
        total_num += len(labels)

        np_labels = labels.cpu().numpy()
        np_loss = loss.detach().cpu().numpy()
        np_pred = pred.detach().cpu().numpy()

        # adv
        # premises_adv, hypotheses_adv = fgsm(premises, hypotheses, pred, model, criterion_all, eps=2e-2, if_infnity=True)
        premises_adv, hypotheses_adv = fgsm(premises, hypotheses, pred, model, criterion_all, eps=1e-1)
        logits_adv, probs_adv, _ = model(premises_adv, hypotheses_adv)

        running_accuracy += correct_predictions(probs, labels)

        # adv_loss = ShannonEntropy(logits_adv, probs)
        adv_loss = criterion(logits_adv, pred)
        np_adv_loss = adv_loss.detach().cpu().numpy()

        running_loss_pos += np_adv_loss[(np_labels==1)].sum() # &(np_pred==0)
        running_loss_neg += np_adv_loss[(np_labels==0)].sum() # &(np_pred==1)

        if batch_index == 0:
            adv_loss_pos = np_adv_loss[np_labels==1]
            adv_loss_neg = np_adv_loss[np_labels==0]
        else:
            adv_loss_pos = np.concatenate((adv_loss_pos, np_adv_loss[(np_labels==1)]), axis=0)
            adv_loss_neg = np.concatenate((adv_loss_neg, np_adv_loss[(np_labels==0)]), axis=0)

    epoch_time = time.time() - epoch_start
    epoch_accuracy = running_accuracy / total_num
    print(running_loss_pos, running_loss_neg)

    losses = np.concatenate((adv_loss_pos, adv_loss_neg), axis=0)
    labels = np.concatenate((np.ones_like(adv_loss_pos), np.zeros_like(adv_loss_neg)), axis=0)
    auc_score = roc_auc(labels, losses)
    creterion_func(adv_loss_pos, adv_loss_neg)
    print('[ROC_AUC] score: %.2f%%' % (100. * auc_score))

    return epoch_time, epoch_accuracy
