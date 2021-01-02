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

    # Switch to evaluate mode.
    adv_loss_entailment, adv_loss_neutral, adv_loss_contradiction = None, None, None

    model.train()
    device = model.device

    epoch_start = time.time()
    running_loss = 0.0
    running_accuracy = 0.0
    total_num = 0

    # Deactivate autograd for evaluation.
    for batch_index, batch in enumerate(dataloader):
        premises = batch["premise"].to(device)
        premises_lengths = batch["premise_length"].to(device)
        hypotheses = batch["hypothesis"].to(device)
        hypotheses_lengths = batch["hypothesis_length"].to(device)
        labels = batch["label"].to(device)

        logits, probs, _, embed = model(premises, premises_lengths, hypotheses, hypotheses_lengths)
        preds = torch.argmax(probs, dim=1)

        premises_adv, hypotheses_adv = fgsm_esim(embed[0], embed[1], preds, model, criterion_all,
                                                 premises_lengths, hypotheses_lengths, embed[2], embed[3],
                                                 eps=5e-2, if_infnity=True)
        logits_adv, probs_adv, _, _ = model(premises_adv, premises_lengths, hypotheses_adv, hypotheses_lengths,
                                       True, embed[2], embed[3])

        loss = criterion(logits, labels)
        running_loss += loss.sum().item()
        total_num += len(labels)

        np_labels = labels.cpu().numpy()
        np_loss = loss.detach().cpu().numpy()

        running_accuracy += correct_predictions(probs, labels)
        # adv_loss = ShannonEntropy(logits_adv, probs)
        adv_loss = criterion(logits_adv, preds)
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

        # if batch_index == 20:
        #     break

    epoch_time = time.time() - epoch_start
    epoch_accuracy = running_accuracy / total_num
    print(np.mean(adv_loss_entailment), np.mean(adv_loss_neutral), np.mean(adv_loss_contradiction))

    # losses = np.concatenate((adv_loss_pos, adv_loss_neg), axis=0)
    # labels = np.concatenate((np.ones_like(adv_loss_pos), np.zeros_like(adv_loss_neg)), axis=0)
    # auc_score = roc_auc(labels, losses)
    creterion_func(adv_loss_entailment, adv_loss_neutral, adv_loss_contradiction, loss_num_respectively=300)
    # print('[ROC_AUC] score: %.2f%%' % (100. * auc_score))

    return epoch_time, epoch_accuracy
