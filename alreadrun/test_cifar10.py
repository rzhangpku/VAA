import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torchvision
import torchvision.transforms as transforms
import os
import torch
import torch.nn.functional as F
from a3v.droped.resnet import PreActResNet18
from a3v.droped.resnet_top import PreActResNet18Top
from torch.autograd import Variable
import sys
from utils.utils_base import creterion_cifar
from utils.utils_base import *
import pandas as pd


def fgsm(x, y, net, eps=4 / 255, x_val_min=0, x_val_max=1):
    x_adv = Variable(x, requires_grad=True)
    h_adv, _ = net(x_adv)
    cost = criterion(h_adv, y)

    net.zero_grad()
    if x_adv.grad is not None:
        x_adv.grad.data.fill_(0)
    cost.backward()

    x_adv = x_adv + eps * x_adv.grad.sign_()
    x_adv = torch.clamp(x_adv, x_val_min, x_val_max)
    return x_adv


def test(epoch):
    global best_acc
    net[0].eval()
    test_loss = 0
    correct = 0
    total = 0

    criterion_losses, all_label = None, None

    criterion = nn.CrossEntropyLoss(reduction='none')
    for batch_idx, (inputs, targets) in enumerate(testloader):
        inputs, targets = inputs.to(device), targets.to(device)
        out, vulner_logits = net[0](inputs)
        preds = torch.argmax(out, dim=1)
        x_adv = fgsm(inputs, preds, net[0])
        _, logits_adv = net[0](x_adv)

        # adv_loss = ShannonEntropy(logits_adv, F.softmax(out,dim=-1))
        adv_loss = criterion(logits_adv, preds)
        np_adv_loss = adv_loss.detach().cpu().numpy()
        np_labels = targets.cpu().numpy()

        total += targets.size(0)
        correct += preds.eq(targets).sum().item()

        if batch_idx == 0:
            criterion_losses = np_adv_loss
            all_label = np_labels
        else:
            criterion_losses = np.concatenate((criterion_losses, np_adv_loss), axis=0)
            all_label = np.concatenate((all_label, np_labels), axis=0)

    creterion_cifar(criterion_losses, all_label)
    data = pd.DataFrame()
    data['loss'] = criterion_losses
    data['label'] = all_label
    data.to_csv('cifar.csv')
    # Save checkpoint.
    acc = 100.*correct/total
    print('valid acc: %.2f' % acc)
    if acc > best_acc:
        print('update resnet ckpt!')
        state = {
            'net': net[1].state_dict(),
            'acc': acc,
            'epoch': epoch,
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(state, CLASSIFY_CKPT_TOP)
        best_acc = acc


if __name__ == '__main__':
    LOAD_CKPT = False
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    BATCH_SIZE = 32
    CLASSIFY_CKPT = './data/checkpoints/cifar/resnet.pth'
    CLASSIFY_CKPT_TOP = './data/checkpoints/cifar/resnet_top.pth'

    best_acc = 0  # best test accuracy
    start_epoch = 0  # start from epoch 0 or last checkpoint epoch

    # Data
    print('==> Preparing data..')
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
    ])

    testset = torchvision.datasets.CIFAR10(root='./data/dataset/cifar10', train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    # Model
    print('==> Building model..')
    net = []
    net.append(PreActResNet18().to(device))

    if device == 'cuda':
        net[0] = torch.nn.DataParallel(net[0])
        cudnn.benchmark = True

    print('==> Resuming from checkpoint..')
    checkpoint = torch.load(CLASSIFY_CKPT)
    net[0].load_state_dict(checkpoint['net'])
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch']

    criterion = nn.CrossEntropyLoss()

    test(0)

