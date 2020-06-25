import torch
import math
from torch.nn import CrossEntropyLoss
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.data import dataloader
from torchvision import transforms
from torchvision.datasets import MNIST, CIFAR10
import pytorch_optmizers
import torch.nn.functional as F
from torch import nn, optim
import numpy as np
from resnet_cifar import resnet20_cifar


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


class CIFAR_Net(nn.Module):
    def __init__(self):
        super(CIFAR_Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class CONV_AE_MNIST(nn.Module):
    def __init__(self):
        super(CONV_AE_MNIST, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, 3, stride=3, padding=1),  # b, 16, 10, 10
            nn.Softplus(True),
            nn.MaxPool2d(2, stride=2),  # b, 16, 5, 5
            nn.Conv2d(16, 8, 3, stride=2, padding=1),  # b, 8, 3, 3
            nn.Softplus(True),
            nn.MaxPool2d(2, stride=1)  # b, 8, 2, 2
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(8, 16, 3, stride=2),  # b, 16, 5, 5
            nn.Softplus(True),
            nn.ConvTranspose2d(16, 8, 5, stride=3, padding=1),  # b, 8, 15, 15
            nn.Softplus(True),
            nn.ConvTranspose2d(8, 1, 2, stride=2, padding=1),  # b, 1, 28, 28
            nn.Tanh()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


class AE_MNIST(nn.Module):
    def __init__(self):
        super(AE_MNIST, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(28 * 28, 512),
            nn.Softplus(),
            nn.Linear(512, 256),
            nn.Softplus(),
            nn.Linear(256, 128),
            nn.Softplus(),
            nn.Linear(128, 32),
            nn.Softplus()
        )
        self.decoder = nn.Sequential(
            nn.Linear(32, 128),
            nn.Softplus(),
            nn.Linear(128, 256),
            nn.Softplus(),
            nn.Linear(256, 512),
            nn.Softplus(),
            nn.Linear(512, 28 * 28),
            nn.Sigmoid())

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

    def reset_parameters(self):
        #nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        fan_in, fan_out = nn.init._calculate_fan_in_and_fan_out(self.weight)
        bound = math.sqrt(6 / (fan_in + fan_out))
        if self.bias is not None:
            nn.init.uniform_(self.bias, -bound, bound)
        nn.init.uniform_(self.weight, -bound, bound)

def to_img(x):
    #x = 0.5 * (x + 1)
    x = x.clamp(0, 1)
    x = x.view(x.size(0), 1, 28, 28)
    return x


network_to_use = 'AE_MNIST'  # AE_MNIST, CNN_MNIST, CONV_AE_MNIST, CNN_CIFAR

transforms_dict = {
        'CNN_MNIST': transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ]),
        'AE_MNIST': transforms.Compose([
                           transforms.ToTensor(),
                           #transforms.Normalize((0.5,), (0.5,))
                       ]),
        'CONV_AE_MNIST': transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.5,), (0.5,))
                       ]),
        'CNN_CIFAR': transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    }

models = {'AE_MNIST': AE_MNIST(),
          'CONV_AE_MNIST': CONV_AE_MNIST(),
          'CNN_MNIST': Net(),
          'CNN_CIFAR': CIFAR_Net()}

criteria = {'AE_MNIST': nn.MSELoss(reduction='mean'),
            'CONV_AE_MNIST': nn.MSELoss(reduction='mean'),
            'CNN_MNIST': F.nll_loss,
            'CNN_CIFAR': nn.CrossEntropyLoss()}


# Loading the data
def dataset(network_to_use_):
    if 'MNIST' in network_to_use_:
        return MNIST
    elif 'CIFAR' in network_to_use_:
        return CIFAR10

dataset_ = dataset(network_to_use)('../data', train=True, download=True,
                                   transform=transforms_dict[network_to_use])
n = len(dataset_)
train_size = int(n*(5/6))
train_set, val_set = torch.utils.data.random_split(dataset_, [train_size, n-train_size])

test = dataset(network_to_use)('./data', train=False, download=True,
                               transform=transforms_dict[network_to_use], )


#loss_fn = CrossEntropyLoss()

loss_fn = criteria[network_to_use]
# Loading the model
#model = resnet20_cifar()
#input_size = 784
#hidden_sizes = [128, 64]
#output_size = 10

if torch.cuda.is_available():
    dev = 'cuda'
else:
    dev = 'cpu'

print('Using dev', dev)
model = models[network_to_use].to(dev)

print(model)
# MNIST opt
opt = dict(model=model,
           loss_fn=loss_fn,
           n=n,
           log_interval=100,
           problem=network_to_use,
           subproblem_solver='adaptive',  # adaptive, non-adaptive
           delta_momentum=True,
           delta_momentum_stepsize=0.01,
           initial_penalty_parameter=10,  # 15000, 10
           verbose=False,
           beta_lipschitz=1,
           eta=0.3,
           sample_size_hessian=10,
           sample_size_gradient=100
           )

#
optimizer = pytorch_optmizers.SRC(model.parameters(), opt=opt)
#optimizer = optim.SGD(model.parameters(), lr=0.01)
#scheduler = MultiStepLR(optimizer, [81, 122, 164], gamma=0.1)

# Detecting device

optimizer.defaults['dev'] = dev
# ToDo: can we do so? (We increase the sample size if case 1 is not satisfied)
optimizer.defaults['double_sample_size'] = False
# Sampling schemes
#exp_growth_constant_grad = ((1-optimizer.defaults['sample_size_gradient'])*n)**(1/optimizer.defaults['n_iterations'])

sampling_scheme = dict(fixed_grad=int(optimizer.defaults['sample_size_gradient']),

                       #exponential_grad=lambda iter_: int(
                       #    min(n, n * optimizer.defaults['sample_size_gradient'] +
                       #        exp_growth_constant_grad**(iter_ + 1))),
                       fixed_hess=int(optimizer.defaults['sample_size_hessian']),
                       #exponential_hess=lambda iter_: int(
                       #    min(n, n * optimizer.defaults['sample_size_hessian'] +
                       #        exp_growth_constant_grad**(iter_ + 1))),
                       )


def init_train_loader(dataloader_, train_, sampling_scheme_name='fixed_grad', n_points_=None):
    n_points = n_points_ if n_points_ else sampling_scheme[sampling_scheme_name]
    print('Loaded ', n_points, 'data points')
    dataloader_args = dict(shuffle=True, batch_size=n_points, num_workers=0)
    train_loader = dataloader_.DataLoader(train_, **dataloader_args)
    dataloader_args['batch_size'] = 1000
    return dataloader_args, train_loader


# Init train loader
dataloader_args, train_loader = init_train_loader(dataloader, train_set)
test_loader_ = dataloader.DataLoader(test, **dataloader_args)
val_loader = dataloader.DataLoader(val_set, **dataloader_args)

test_loader = (test_loader_, val_loader)

_, train_loader_hess = init_train_loader(dataloader, train_set, sampling_scheme_name='fixed_hess')


#arr = torch.from_numpy(np.load('test_vec.npy')).view(-1)
#print(loss_fn(model(arr), arr))
#dataiter = iter(train_loader)
#images = dataiter.next()
#arr = images[0][0][0].detach().numpy()
#print(arr)
#np.save('test_vec.npy', arr)

