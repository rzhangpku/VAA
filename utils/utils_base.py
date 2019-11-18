from torch.autograd import Variable
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
import torch
import pandas as pd
import numpy as np
import torch.nn.functional as F

def roc_auc(labels, losses):
    fpt, tpt, thresholds = roc_curve(labels, losses)
    roc_auc = auc(fpt, tpt)
    plt.switch_backend('Agg')
    fig = plt.figure()
    lw = 0.8
    plt.plot(fpt, tpt, color='red',
             lw=lw, label='ROC curve (auc = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='blue', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    # plt.title('adversarial detect roc curve')
    plt.legend(loc="lower right")
    fig.savefig('roc.png', dpi=500)
    return roc_auc

def creterion_func(entailment, neutral, contradiction=None, loss_num_respectively=400):
    fig = plt.figure(figsize=(8,6))
    entailment = entailment[:loss_num_respectively]
    neutral = neutral[:loss_num_respectively]
    # creterion = pd.DataFrame([benign_losses, adv_losses])
    # creterion.to_csv('creterion.csv', index=False)
    if contradiction is not None:
        contradiction = contradiction[:loss_num_respectively]
        plt.scatter(np.arange(len(entailment)), entailment, color='cornflowerblue', s=3, marker='o', label='entailment')
        plt.scatter(np.arange(len(neutral)), neutral, color='crimson', s=3, marker='*',  label='neutral')
        plt.scatter(np.arange(len(contradiction)), contradiction, color='green', s=3, marker='^',  label='contradiction')
        plt.plot([0, len(contradiction) - 1], [np.mean(contradiction), np.mean(contradiction)],
                 color='green', linestyle='--')
    else:
        plt.scatter(np.arange(len(entailment)), entailment, color='cornflowerblue', s=3, marker='o', label='paraphrase')
        plt.scatter(np.arange(len(neutral)), neutral, color='crimson', s=3, marker='*', label='non-paraphrase')

    plt.plot([0, len(entailment)-1], [np.mean(entailment), np.mean(entailment)], color='cornflowerblue', linestyle='--')
    plt.plot([0, len(neutral) - 1], [np.mean(neutral), np.mean(neutral)], color='crimson', linestyle='--')
    # plt.xticks([])
    plt.xlabel('Sentence pair')
    plt.ylabel('Criterion loss')
    plt.legend(loc="upper right")
    fig.savefig('creterion.png', dpi=500)
    plt.show()

def creterion_cifar(criterion_losses, all_label, loss_num=200):
    fig = plt.figure(figsize=(8,6))
    # 'plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck'
    temp = criterion_losses[all_label==0][:loss_num]
    plt.scatter(np.arange(loss_num), temp, color='darkviolet', s=3, marker='1', label='plane')
    plt.plot([0, loss_num-1], [np.mean(temp), np.mean(temp)], color='darkviolet', linestyle='--')

    temp = criterion_losses[all_label == 1][:loss_num]
    plt.scatter(np.arange(loss_num), temp, color='yellow', s=3, marker='2',  label='car')
    plt.plot([0, loss_num - 1], [np.mean(temp), np.mean(temp)], color='yellow', linestyle='--')

    temp = criterion_losses[all_label == 2][:loss_num]
    plt.scatter(np.arange(loss_num), temp, color='aqua', s=3, marker='3',  label='bird')
    plt.plot([0, loss_num - 1], [np.mean(temp), np.mean(temp)],
             color='aqua', linestyle='--')

    temp = criterion_losses[all_label == 3][:loss_num]
    plt.scatter(np.arange(loss_num), temp, color='darkslateblue', s=3, marker='4',  label='cat')
    plt.plot([0, loss_num - 1], [np.mean(temp), np.mean(temp)],
             color='darkslateblue', linestyle='--')

    temp = criterion_losses[all_label == 4][:loss_num]
    plt.scatter(np.arange(loss_num), temp, color='tan', s=3, marker='o',  label='deer')
    plt.plot([0, loss_num - 1], [np.mean(temp), np.mean(temp)],
             color='tan', linestyle='--')

    temp = criterion_losses[all_label == 5][:loss_num]
    plt.scatter(np.arange(loss_num), temp, color='lightgreen', s=3, marker='v',  label='dog')
    plt.plot([0, loss_num - 1], [np.mean(temp), np.mean(temp)],
             color='lightgreen', linestyle='--')

    temp = criterion_losses[all_label == 6][:loss_num]
    plt.scatter(np.arange(loss_num), temp, color='violet', s=3, marker='*',  label='frog')
    plt.plot([0, loss_num - 1], [np.mean(temp), np.mean(temp)],
             color='violet', linestyle='--')

    temp = criterion_losses[all_label == 7][:loss_num]
    plt.scatter(np.arange(loss_num), temp, color='steelblue', s=3, marker='<',  label='horse')
    plt.plot([0, loss_num - 1], [np.mean(temp), np.mean(temp)],
             color='steelblue', linestyle='--')

    temp = criterion_losses[all_label == 8][:loss_num]
    plt.scatter(np.arange(loss_num), temp, color='darkred', s=3, marker='>',  label='ship')
    plt.plot([0, loss_num - 1], [np.mean(temp), np.mean(temp)],
             color='darkred', linestyle='--')

    temp = criterion_losses[all_label == 9][:loss_num]
    plt.scatter(np.arange(loss_num), temp, color='salmon', s=3, marker='^',  label='truck')
    plt.plot([0, loss_num - 1], [np.mean(temp), np.mean(temp)],
             color='salmon', linestyle='--')

    # plt.xticks([])
    plt.xlabel('Image')
    plt.ylabel('Criterion loss')
    plt.legend(loc="upper right")
    fig.savefig('creterion.png', dpi=500)
    plt.show()

def fgsm(premises, hypotheses, y, net, criterion, eps=2e-2, if_infnity=False):
    premises_adv = Variable(premises, requires_grad=True)
    hypotheses_adv = Variable(hypotheses, requires_grad=True)

    premises_mask = (torch.sum(premises, dim=-1) != 0).float().unsqueeze(-1)
    hypotheses_mask = (torch.sum(hypotheses, dim=-1) != 0).float().unsqueeze(-1)

    h_adv, _, _ = net(premises_adv, hypotheses_adv)
    cost = criterion(h_adv, y)

    net.zero_grad()
    if premises_adv.grad is not None:
        premises_adv.grad.data.fill_(0)
    if hypotheses_adv.grad is not None:
        hypotheses_adv.grad.data.fill_(0)
    cost.backward()

    if if_infnity:
        # L无穷
        premises_adv = premises_adv + eps * premises_adv.grad.sign_()
        hypotheses_adv = hypotheses_adv + eps * hypotheses_adv.grad.sign_()
    else:
        # L2
        premises_adv = premises_adv + eps * premises_adv.grad / torch.norm(premises_adv.grad, dim=(1,2), keepdim=True)
        hypotheses_adv = hypotheses_adv + eps * hypotheses_adv / torch.norm(hypotheses_adv.grad, dim=(1,2), keepdim=True)

    premises_adv = premises_adv * premises_mask
    hypotheses_adv = hypotheses_adv * hypotheses_mask

    return premises_adv, hypotheses_adv

def fgsm_esim(premises, hypotheses, y, net, criterion, premises_lengths, hypotheses_lengths,
              premises_mask, hypotheses_mask, eps=2e-2, if_infnity=False):
    premises_adv = Variable(premises, requires_grad=True)
    hypotheses_adv = Variable(hypotheses, requires_grad=True)

    _premises_mask = (torch.sum(premises, dim=-1) != 0).float().unsqueeze(-1)
    _hypotheses_mask = (torch.sum(hypotheses, dim=-1) != 0).float().unsqueeze(-1)

    h_adv, _, _, _ = net(premises_adv, premises_lengths, hypotheses_adv, hypotheses_lengths,
                                    True, premises_mask, hypotheses_mask)
    cost = criterion(h_adv, y)

    net.zero_grad()
    if premises_adv.grad is not None:
        premises_adv.grad.data.fill_(0)
    if hypotheses_adv.grad is not None:
        hypotheses_adv.grad.data.fill_(0)
    cost.backward()

    if if_infnity:
        # L无穷
        premises_adv = premises_adv + eps * premises_adv.grad.sign_()
        hypotheses_adv = hypotheses_adv + eps * hypotheses_adv.grad.sign_()
    else:
        # L2
        premises_adv = premises_adv + eps * premises_adv.grad / torch.norm(premises_adv.grad, dim=(1,2), keepdim=True)
        hypotheses_adv = hypotheses_adv + eps * hypotheses_adv / torch.norm(hypotheses_adv.grad, dim=(1,2), keepdim=True)

    premises_adv = premises_adv * _premises_mask
    hypotheses_adv = hypotheses_adv * _hypotheses_mask

    return premises_adv, hypotheses_adv


def jacobian(premises, hypotheses, y, net, criterion, drop=1):
    """
    选择影响最大词向量的时候应该求二范数
    :param premises:
    :param hypotheses:
    :param y:
    :param net:
    :param criterion:
    :param drop:
    :return:
    """
    premises_adv = Variable(premises, requires_grad=True)
    hypotheses_adv = Variable(hypotheses, requires_grad=True)
    h_adv, _, _ = net(premises_adv, hypotheses_adv)

    for i in range(drop):
        grads = None
    return premises_adv, hypotheses_adv

def ShannonEntropy(logits, soft_label):
    log_probs = F.log_softmax(logits, dim=-1)
    H = - torch.sum(torch.mul(soft_label, log_probs), dim=-1)
    return H

