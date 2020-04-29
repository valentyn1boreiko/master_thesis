import torch
from torch.nn import CrossEntropyLoss
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.data import dataloader
from torchvision import transforms
from torchvision.datasets import CIFAR10
import pytorch_optmizers
from resnet_cifar import resnet20_cifar


# Loading the data
train = CIFAR10('../2nd-order/data/', train=True, download=True, transform=transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),  # ToTensor does min-max normalization.
]), )

# Get the number of samples in the dataset
n = train.data.shape[0]

test = CIFAR10('../2nd-order/data/', train=False, download=True, transform=transforms.Compose([
    transforms.ToTensor(),  # ToTensor does min-max normalization.
]), )

# Loading the model
model = resnet20_cifar()
optimizer = pytorch_optmizers.SRC(model.parameters())
#scheduler = MultiStepLR(optimizer, [81, 122, 164], gamma=0.1)
loss_fn = CrossEntropyLoss()

# Detecting device
if torch.cuda.is_available():
    dev = 'cuda'
    model = model.cuda()
else:
    dev = 'cpu'

# Sampling schemes
exp_growth_constant_grad = ((1-optimizer.defaults['sample_size_gradient'])*n)**(1/optimizer.defaults['n_iterations'])

sampling_scheme = dict(fixed=int(n * optimizer.defaults['sample_size_gradient']),
                       exponential=lambda iter_: int(
                           min(n, n * optimizer.defaults['sample_size_gradient'] +
                               exp_growth_constant_grad**(iter_ + 1))
                       )
                       )


def init_train_loader(dataloader, train, sampling_scheme_name='fixed'):
    dataloader_args = dict(shuffle=True, batch_size=sampling_scheme[sampling_scheme_name], num_workers=4)
    train_loader = dataloader.DataLoader(train, **dataloader_args)
    return dataloader_args, train_loader


# Init train loader
dataloader_args, train_loader = init_train_loader(dataloader, train)
test_loader = dataloader.DataLoader(test, **dataloader_args)
