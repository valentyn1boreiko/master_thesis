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
import autograd_hacks
import numpy as np
from resnet_cifar import resnet20_cifar

torch.manual_seed(7)
torch.set_printoptions(precision=10)


class Flatten(nn.Module):

    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, x):
        shape = torch.prod(torch.tensor(x.shape[1:])).item()
        return x.view(-1, shape)


def LinearRegression_():
    model_ = nn.Sequential(
        Flatten(),
        nn.Linear(784, 10, bias=True),
    )
    return model_


#LinearRegression_ = nn.Sequential()
#LinearRegression_.add_module('flat', Flatten())
#LinearRegression_.add_module('weights', nn.Linear(784, 10, bias=True))


class LinearRegression(torch.nn.Module):
    def __init__(self):
        super(LinearRegression, self).__init__()
        self.out = nn.Sequential(
            Flatten(),
            nn.Linear(784, 10)
        )

    def forward(self, x):
        out = self.out(x)
        return out


# Source - https://github.com/pytorch/examples/blob/master/mnist/main.py


class Net(nn.Module):
    def __init__(self, activation_='swish'):
        super(Net, self).__init__()
        activation_dict = {
            'relu': F.relu,
            'softplus': F.softplus,
            'swish': lambda x: x * torch.sigmoid(x),
        }
        self.activation = activation_dict[activation_]
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout2d(0.25)
        self.dropout2 = nn.Dropout2d(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.activation(x)
        x = self.conv2(x)
        x = self.activation(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = self.activation(x)
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
    def __init__(self, activation_='softplus'):
        super(AE_MNIST, self).__init__()
        activation_dict = {
            'relu': nn.ReLU,
            'softplus': nn.Softplus,
            # ToDo: correct with the wrapper!
            'swish': nn.Sigmoid
        }
        self.activation = activation_dict[activation_]
        self.encoder = nn.Sequential(
            nn.Linear(28 * 28, 512),
            self.activation(),
            nn.Linear(512, 256),
            self.activation(),
            nn.Linear(256, 128),
            self.activation(),
            nn.Linear(128, 32),
            self.activation()
        )
        self.decoder = nn.Sequential(
            nn.Linear(32, 128),
            self.activation(),
            nn.Linear(128, 256),
            self.activation(),
            nn.Linear(256, 512),
            self.activation(),
            nn.Linear(512, 28 * 28),
            nn.Sigmoid())

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


# Uniform Glorot initializer
def weights_init(m):
    if isinstance(m, nn.Linear):
        #nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        fan_in, fan_out = nn.init._calculate_fan_in_and_fan_out(m.weight)
        bound = math.sqrt(6 / (fan_in + fan_out))
        nn.init.uniform_(m.bias, -bound, bound)
        nn.init.uniform_(m.weight, -bound, bound)

def to_img(x):
    #x = 0.5 * (x + 1)
    x = x.clamp(0, 1)
    x = x.view(x.size(0), 1, 28, 28)
    return x


network_to_use = 'AE_MNIST'  # AE_MNIST, CNN_MNIST, CONV_AE_MNIST, CNN_CIFAR, LIN_REG_MNIST
activation = 'softplus'  # swish, softplus, relu

transforms_dict = {
        'CNN_MNIST': transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ]),
        'AE_MNIST': transforms.Compose([
                           transforms.ToTensor(),
                           #transforms.Normalize((0.5,), (0.5,))
                       ]),
        'LIN_REG_MNIST': transforms.Compose([
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

models = {'AE_MNIST': AE_MNIST(activation),
          'CONV_AE_MNIST': CONV_AE_MNIST(),
          'CNN_MNIST': Net(activation),
          'CNN_CIFAR': CIFAR_Net(),
          'LIN_REG_MNIST': LinearRegression()}

criteria = {'AE_MNIST': nn.MSELoss(reduction='mean'),
            'CONV_AE_MNIST': nn.MSELoss(reduction='mean'),
            'CNN_MNIST': F.nll_loss,
            'CNN_CIFAR': nn.CrossEntropyLoss(),
            'LIN_REG_MNIST': nn.MSELoss(reduction='mean')}  # nn.CrossEntropyLoss(reduction='mean')}


# Loading the data
def dataset(network_to_use_):
    if 'MNIST' in network_to_use_:
        return MNIST
    elif 'CIFAR' in network_to_use_:
        return CIFAR10

torch.manual_seed(7)

dataset_ = dataset(network_to_use)('../data', train=True, download=True,
                                   transform=transforms_dict[network_to_use])
n = len(dataset_)
train_size = int(n*(5/6))
train_set, val_set = torch.utils.data.random_split(dataset_, [train_size, n-train_size])

test = dataset(network_to_use)('../data', train=False, download=True,
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
start_model_path = None
# "models/benchmarks_AE_MNIST_Adam_5.pt"

model = models[network_to_use].to(dev)

#autograd_hacks.add_hooks(model)

torch.manual_seed(7)
if start_model_path:
    model.load_state_dict(torch.load(start_model_path))
else:
    model.apply(weights_init)

print(model)
# MNIST opt
opt = dict(model=model,
           loss_fn=loss_fn,
           activation=activation,
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
           sample_size_hessian=100,
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

torch.manual_seed(7)
def init_train_loader(dataloader_, train_, sampling_scheme_name='fixed_grad', n_points_=None):
    n_points = n_points_ if n_points_ else sampling_scheme[sampling_scheme_name]
    print('Loaded ', n_points, 'data points')
    dataloader_args = dict(shuffle=True, batch_size=n_points, num_workers=0)
    train_loader = dataloader_.DataLoader(train_, **dataloader_args)
    return dataloader_args, train_loader



#arr = torch.from_numpy(np.load('test_vec.npy')).view(-1)
#print(loss_fn(model(arr), arr))
#dataiter = iter(val_loader)
#images = dataiter.next()
#print(len(test_loader_.dataset), int(optimizer.defaults['sample_size_gradient']), images[0].sum())
#exit(0)
#arr = images[0][0][0].detach().numpy()
#print(arr)
#np.save('test_vec.npy', arr)

