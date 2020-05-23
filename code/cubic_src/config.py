import torch
from torch.nn import CrossEntropyLoss
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.data import dataloader
from torchvision import transforms
from torchvision.datasets import CIFAR10
from torchvision.datasets import MNIST
import pytorch_optmizers
import torch.nn.functional as F
from torch import nn, optim
from resnet_cifar import resnet20_cifar

problem = 'MNIST'  # MNIST, matrix_completion


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout2d(0.25)
        self.dropout2 = nn.Dropout2d(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output


# Loading the data
train = MNIST('./data', train=True, download=True, transform=transforms.Compose([
                               transforms.ToTensor(),
                               transforms.Normalize(
                                 (0.1307,), (0.3081,))
                             ]), )

# Get the number of samples in the dataset
n = train.data.shape[0]

test = MNIST('./data', train=False, download=True, transform=transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(
        (0.1307,), (0.3081,))
]), )

#loss_fn = CrossEntropyLoss()

loss_fn = F.nll_loss
# Loading the model
#model = resnet20_cifar()
input_size = 784
hidden_sizes = [128, 64]
output_size = 10

if torch.cuda.is_available():
    dev = 'cuda'
else:
    dev = 'cpu'

model = Net().to(dev)

# MNIST opt
opt = dict(model=model,
           loss_fn=loss_fn,
           n=n,
           log_interval=3)

#
optimizer = pytorch_optmizers.SRC(model.parameters(), opt=opt)
#scheduler = MultiStepLR(optimizer, [81, 122, 164], gamma=0.1)

# Detecting device


optimizer.defaults['dev'] = dev
# ToDo: can we do so? (We increase the sample size if case 1 is not satisfied)
optimizer.defaults['double_sample_size'] = False
# Sampling schemes
#exp_growth_constant_grad = ((1-optimizer.defaults['sample_size_gradient'])*n)**(1/optimizer.defaults['n_iterations'])

sampling_scheme = dict(fixed_grad=int(n * optimizer.defaults['sample_size_gradient']),

                       #exponential_grad=lambda iter_: int(
                       #    min(n, n * optimizer.defaults['sample_size_gradient'] +
                       #        exp_growth_constant_grad**(iter_ + 1))),
                       fixed_hess=int(n * optimizer.defaults['sample_size_hessian']),
                       #exponential_hess=lambda iter_: int(
                       #    min(n, n * optimizer.defaults['sample_size_hessian'] +
                       #        exp_growth_constant_grad**(iter_ + 1))),
                       )


def init_train_loader(dataloader_, train_, sampling_scheme_name='fixed_grad', n_points_=None):
    n_points = n_points_ if n_points_ else sampling_scheme[sampling_scheme_name]
    print('Loaded ', n_points, 'data points')
    dataloader_args = dict(shuffle=True, batch_size=n_points, num_workers=4)
    train_loader = dataloader_.DataLoader(train_, **dataloader_args)
    dataloader_args['batch_size'] = 1000
    return dataloader_args, train_loader


# Init train loader
dataloader_args, train_loader = init_train_loader(dataloader, train)
test_loader = dataloader.DataLoader(test, **dataloader_args)

_, train_loader_hess = init_train_loader(dataloader, train, sampling_scheme_name='fixed_hess')

