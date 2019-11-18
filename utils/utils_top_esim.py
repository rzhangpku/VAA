"""
Utility functions for training and validating models.
"""

import time
import torch.nn as nn
from tqdm import tqdm
from mfae.utils import correct_predictions
from utils.utils_base import *

def train(model,
          dataloader,
          optimizer,
          criterion,
          epoch_number,
          max_gradient_norm):
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
    model[0].train()
    model[1].train()
    device = model[0].device

    epoch_start = time.time()
    batch_time_avg = 0.0
    running_loss = 0.0
    correct_preds = 0

    tqdm_batch_iterator = tqdm(dataloader)
    for batch_index, batch in enumerate(tqdm_batch_iterator):
        batch_start = time.time()

        # Move input and output data to the GPU if it is used.
        premises = batch["premise"].to(device)
        premises_lengths = batch["premise_length"].to(device)
        hypotheses = batch["hypothesis"].to(device)
        hypotheses_lengths = batch["hypothesis_length"].to(device)
        labels = batch["label"].to(device)

        _, probabilities, esim_logits, embed = model[0](premises, premises_lengths, hypotheses, hypotheses_lengths)
        preds = torch.argmax(probabilities, dim=1)
        premises_adv, hypotheses_adv = fgsm_esim(embed[0], embed[1], preds, model[0], criterion,
                                                 premises_lengths, hypotheses_lengths, embed[2], embed[3],
                                                 eps=2e-2, if_infnity=False)
        _, _, adv_logits, _ = model[0](premises_adv, premises_lengths, hypotheses_adv, hypotheses_lengths,
                                    True, embed[2], embed[3])
        vulnerability = torch.cat([esim_logits - adv_logits, esim_logits, adv_logits], dim=1)

        logits, probs = model[1](premises_adv, premises_lengths, hypotheses_adv, hypotheses_lengths, vulnerability,
                                 embed[2],embed[3])

        optimizer.zero_grad()
        loss = criterion(logits, labels)
        loss.backward()

        # nn.utils.clip_grad_norm_(model[0].parameters(), max_gradient_norm)
        nn.utils.clip_grad_norm_(model[1].parameters(), max_gradient_norm)
        optimizer.step()

        batch_time_avg += time.time() - batch_start
        running_loss += loss.item()
        correct_preds += correct_predictions(probs, labels)

        description = "Avg. batch proc. time: {:.4f}s, loss: {:.4f}"\
                      .format(batch_time_avg/(batch_index+1),
                              running_loss/(batch_index+1))
        tqdm_batch_iterator.set_description(description)

    epoch_time = time.time() - epoch_start
    epoch_loss = running_loss / len(dataloader)
    epoch_accuracy = correct_preds / len(dataloader.dataset)

    return epoch_time, epoch_loss, epoch_accuracy


def validate(model, dataloader, criterion):
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
    model[0].train()
    model[1].eval()
    device = model[0].device

    epoch_start = time.time()
    running_loss = 0.0
    running_accuracy = 0.0

    # Deactivate autograd for evaluation.
    for batch in dataloader:
        # Move input and output data to the GPU if one is used.
        premises = batch["premise"].to(device)
        premises_lengths = batch["premise_length"].to(device)
        hypotheses = batch["hypothesis"].to(device)
        hypotheses_lengths = batch["hypothesis_length"].to(device)
        labels = batch["label"].to(device)

        _, probabilities, esim_logits, embed = model[0](premises, premises_lengths, hypotheses, hypotheses_lengths)
        preds = torch.argmax(probabilities, dim=1)
        premises_adv, hypotheses_adv = fgsm_esim(embed[0], embed[1], preds, model[0], criterion,
                                                 premises_lengths, hypotheses_lengths, embed[2], embed[3])
        _, _, adv_logits, _ = model[0](premises_adv, premises_lengths, hypotheses_adv, hypotheses_lengths,
                                    True, embed[2], embed[3])
        vulnerability = torch.cat([esim_logits - adv_logits, esim_logits, adv_logits], dim=1)

        logits, probs = model[1](premises_adv, premises_lengths, hypotheses_adv, hypotheses_lengths, vulnerability,
                                 embed[2], embed[3])
        loss = criterion(logits, labels)

        running_loss += loss.item()
        running_accuracy += correct_predictions(probs, labels)

    epoch_time = time.time() - epoch_start
    epoch_loss = running_loss / len(dataloader)
    epoch_accuracy = running_accuracy / (len(dataloader.dataset))
    return epoch_time, epoch_loss, epoch_accuracy


def test(model, dataloader, criterion):
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
    model[0].train()
    model[1].eval()
    device = model[0].device

    data =None
    # Deactivate autograd for evaluation.
    for batch in dataloader:
        # Move input and output data to the GPU if one is used.
        premises = batch["premise"].to(device)
        premises_lengths = batch["premise_length"].to(device)
        hypotheses = batch["hypothesis"].to(device)
        hypotheses_lengths = batch["hypothesis_length"].to(device)
        labels = batch["label"].to(device)

        _, probabilities, esim_logits, embed = model[0](premises, premises_lengths, hypotheses, hypotheses_lengths)
        preds = torch.argmax(probabilities, dim=1)
        premises_adv, hypotheses_adv = fgsm_esim(embed[0], embed[1], preds, model[0], criterion,
                                                 premises_lengths, hypotheses_lengths, embed[2], embed[3])
        _, _, adv_logits, _ = model[0](premises_adv, premises_lengths, hypotheses_adv, hypotheses_lengths,
                                    True, embed[2], embed[3])
        vulnerability = torch.cat([esim_logits - adv_logits, esim_logits, adv_logits], dim=1)

        logits, probs = model[1](premises_adv, premises_lengths, hypotheses_adv, hypotheses_lengths, vulnerability,
                                 embed[2], embed[3])

        pred = torch.argmax(probs, dim=1).cpu().numpy()
        gold_labels = []
        # label_names = ['entailment', 'neutral', 'contradiction']
        for x in pred:
            if x == 0:
                gold_labels.append('entailment')
            elif x == 1:
                gold_labels.append('neutral')
            else:
                gold_labels.append('contradiction')
        temp = pd.DataFrame()
        temp['pairID'] = batch["id"]
        temp['gold_label'] = gold_labels

        if data is None:
            data = temp
        else:
            data = pd.concat((data, temp))
    return data


