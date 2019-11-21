"""
Utility functions for training and validating models.
"""
import time
import torch
import torch.nn as nn
import pandas as pd
from a3v.utils import correct_predictions
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
    running_loss_entailment, running_loss_neutral, running_loss_contradiction = 0.0, 0.0, 0.0
    adv_loss_entailment, adv_loss_neutral, adv_loss_contradiction = None, None, None

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
        # running_accuracy += correct_predictions(probs, labels)
        total_num += len(labels)

        np_labels = labels.cpu().numpy()
        np_loss = loss.detach().cpu().numpy()
        np_pred = pred.detach().cpu().numpy()

        # 'entailment': 0, 'neutral': 1, 'contradiction': 2
        running_loss_entailment += np_loss[(np_labels==0)].sum() # &(np_pred==1)
        running_loss_neutral += np_loss[(np_labels==1)].sum() # &(np_pred==0)
        running_loss_contradiction += np_loss[(np_labels==2)].sum()  # &(np_pred==1)

        # adv
        premises_adv, hypotheses_adv = fgsm(premises, hypotheses, pred, model, criterion_all) # eps=0.05, if_infnity=True
        logits_adv, probs_adv, _ = model(premises_adv, hypotheses_adv)

        running_accuracy += correct_predictions(probs, labels)

        adv_loss = ShannonEntropy(logits_adv, probs)
        # adv_loss = criterion(logits_adv, pred)
        np_adv_loss = adv_loss.detach().cpu().numpy()
        # np_probs_adv = torch.max(probs_adv, dim=1)[0].detach().cpu().numpy()

        if batch_index == 0:
            adv_loss_entailment = np_adv_loss[np_labels==0]
            adv_loss_neutral = np_adv_loss[np_labels==1]
            adv_loss_contradiction = np_adv_loss[np_labels==2]
        else:
            adv_loss_entailment = np.concatenate((adv_loss_entailment, np_adv_loss[(np_labels==0)]), axis=0)
            adv_loss_neutral = np.concatenate((adv_loss_neutral, np_adv_loss[(np_labels==1)]), axis=0)
            adv_loss_contradiction = np.concatenate((adv_loss_contradiction, np_adv_loss[(np_labels==2)]), axis=0)

        # if batch_index == 10:
        #     break

    epoch_time = time.time() - epoch_start
    epoch_accuracy = running_accuracy / total_num
    print(running_loss_entailment, running_loss_neutral, running_loss_contradiction)

    # losses = np.concatenate((adv_loss_pos, adv_loss_neg), axis=0)
    # labels = np.concatenate((np.ones_like(adv_loss_pos), np.zeros_like(adv_loss_neg)), axis=0)
    # auc_score = roc_auc(labels, losses)
    adv_loss_entailment = adv_loss_entailment[adv_loss_entailment<1.5]
    adv_loss_neutral = adv_loss_neutral[adv_loss_neutral < 1.5]
    adv_loss_contradiction = adv_loss_contradiction[adv_loss_contradiction < 1.5]
    creterion_func(adv_loss_entailment, adv_loss_neutral, adv_loss_contradiction)
    # print('[ROC_AUC] score: %.2f%%' % (100. * auc_score))

    return epoch_time, epoch_accuracy
