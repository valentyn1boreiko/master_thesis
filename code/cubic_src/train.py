import torch
from torch.nn import CrossEntropyLoss
from torch.optim import SGD
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.data import dataloader
from torchvision import transforms
from torchvision.datasets import CIFAR10
from pytorch_optmizers import SRC
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

model = resnet20_cifar()
optimizer = SRC(model.parameters())
scheduler = MultiStepLR(optimizer, [81, 122, 164], gamma=0.1)
loss_fn = CrossEntropyLoss()

dataloader_args = dict(shuffle=True, batch_size=optimizer.defaults['sample_size_gradient'], num_workers=4)
train_loader = dataloader.DataLoader(train, **dataloader_args)
test_loader = dataloader.DataLoader(test, **dataloader_args)

if torch.cuda.is_available():
    dev = 'cuda'
    model = model.cuda()
else:
    dev = 'cpu'


def get_accuracy(model, dev, loss_fn, loader):
    model.eval()
    correct_, total_, loss_ = (0, 0, 0)

    for batch_idx_, (data_, target_) in enumerate(loader):
        # Get Samples
        data_ = data_.to(dev)
        target_ = target_.to(dev)
        outputs = model(data_)
        loss_ += loss_fn(outputs, target_).detach() * len(target_)
        # Get prediction
        _, predicted = torch.max(outputs.data, 1)
        # Total number of labels
        total_ += len(target_)
        # Total correct predictions
        correct_ += (predicted == target_).sum().detach()
        del outputs
        del predicted
    acc = 100 * correct_.item() / total_
    loss_ = loss_ / total_
    return loss_.item(), acc


last_model = copy.deepcopy(model)

# We have T_out iterations
for epoch in range(optimizer.defaults['n_iterations']):
    # Set modules in the the network to train mode
    model.train()
    correct, total, loss = (0, 0, 0)
    for batch_idx, (data, target) in enumerate(train_loader):
        optimizer.zero_grad()
        loss_fn(model(data), target).backward(create_graph=True)
