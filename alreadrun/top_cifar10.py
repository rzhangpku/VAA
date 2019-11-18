import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torchvision
import torchvision.transforms as transforms
import os
import torch
from mfae.droped.resnet import PreActResNet18
from mfae.droped.resnet_top import PreActResNet18Top
from torch.autograd import Variable
import sys


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

# Training
def train(epoch):
    print('\nEpoch: %d' % epoch)
    # scheduler.step()
    net[0].train()
    net[1].train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        out, logits = net[0](inputs)
        preds = torch.argmax(out, dim=1)
        x_adv = fgsm(inputs, preds, net[0])
        _, logits_adv = net[0](x_adv)

        vulnerability = torch.cat([logits_adv - logits, logits_adv, logits], dim=1)
        outputs = net[1](inputs, vulnerability)

        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
    acc = 100. * correct / total
    print('train acc: %.2f' % acc)

def test(epoch):
    global best_acc
    net[0].eval()
    net[1].eval()
    test_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(testloader):
        inputs, targets = inputs.to(device), targets.to(device)
        out, logits = net[0](inputs)
        preds = torch.argmax(out, dim=1)
        x_adv = fgsm(inputs, preds, net[0])
        _, logits_adv = net[0](x_adv)

        vulnerability = torch.cat([logits_adv - logits, logits_adv, logits], dim=1)
        outputs = net[1](inputs, vulnerability)

        loss = criterion(outputs, targets)

        test_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

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
    BATCH_SIZE = 128
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

    trainset = torchvision.datasets.CIFAR10(root='./data/dataset/cifar10', train=True, download=True, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)

    testset = torchvision.datasets.CIFAR10(root='./data/dataset/cifar10', train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    # Model
    print('==> Building model..')
    net = []
    net.append(PreActResNet18().to(device))
    net.append(PreActResNet18Top().to(device))

    if device == 'cuda':
        net[0] = torch.nn.DataParallel(net[0])
        net[1] = torch.nn.DataParallel(net[1])
        cudnn.benchmark = True

    print('==> Resuming from checkpoint..')
    checkpoint = torch.load(CLASSIFY_CKPT)
    net[0].load_state_dict(checkpoint['net'])
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch']

    if LOAD_CKPT:
        # Load checkpoint.
        checkpoint = torch.load(CLASSIFY_CKPT_TOP)
        net[1].load_state_dict(checkpoint['net'])
        best_acc = checkpoint['acc']
        start_epoch = checkpoint['epoch']
    else:
        print('load pretrained net1 to net2')
        model_dict = net[1].state_dict()
        pretrained_dict = checkpoint["net"]
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        net[1].load_state_dict(model_dict)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net[1].parameters(), lr=0.0001)
    # scheduler = lr_scheduler.StepLR(optimizer, step_size=150, gamma=0.1)

    for epoch in range(start_epoch, start_epoch+500):
        train(epoch)
        test(epoch)
        sys.stdout.flush()  # 刷新输出

