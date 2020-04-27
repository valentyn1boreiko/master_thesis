import torch
from torch.nn import CrossEntropyLoss
from torch.optim import SGD
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.data import dataloader
from torchvision import transforms
from torchvision.datasets import CIFAR10
import numpy as np
import copy


from resnet_cifar import resnet20_cifar

train = CIFAR10('../2nd-order/data/', train=True, download=True, transform=transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),  # ToTensor does min-max normalization.
]), )

test = CIFAR10('../2nd-order/data/', train=False, download=True, transform=transforms.Compose([
    transforms.ToTensor(),  # ToTensor does min-max normalization.
]), )

dataloader_args = dict(shuffle=True, batch_size=256, num_workers=4)
train_loader = dataloader.DataLoader(train, **dataloader_args)
test_loader = dataloader.DataLoader(test, **dataloader_args)

model = resnet20_cifar()
optimizer = SGD(model.parameters(), lr=1e-1, weight_decay=5e-4, momentum=0.9)
scheduler = MultiStepLR(optimizer, [81, 122, 164], gamma=0.1)
loss_fn = CrossEntropyLoss()

if torch.cuda.is_available():
    dev = 'cuda'
    model = model.cuda()
else:
    dev = 'cpu'


def get_accuracy(model, dev, loss_fn, loader):
    model.eval()
    correct = 0
    total = 0
    loss = 0
    for batch_idx, (data, target) in enumerate(loader):
        # Get Samples
        data = data.to(dev)
        target = target.to(dev)
        outputs = model(data)
        loss += loss_fn(outputs, target).detach() * len(target)
        # Get prediction
        _, predicted = torch.max(outputs.data, 1)
        # Total number of labels
        total += len(target)
        # Total correct predictions
        correct += (predicted == target).sum().detach()
        del outputs
        del predicted
    acc = 100 * correct.item() / total
    loss = loss / total
    return loss.item(), acc


last_model = copy.deepcopy(model)