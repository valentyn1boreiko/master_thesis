import torch
import math
from torch.utils.data import dataloader
from torchvision import transforms
from torchvision.datasets import MNIST, CIFAR10
import pytorch_optmizers
import torch.nn.functional as F
from torch import nn
torch.manual_seed(7)
torch.set_printoptions(precision=10)


# ResNet
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion *
                               planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512*block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


def ResNet18():
    return ResNet(BasicBlock, [2, 2, 2, 2])
# ResNet


# Linear regression
class Flatten(nn.Module):

    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, x):
        shape = torch.prod(torch.tensor(x.shape[1:])).item()
        return x.view(-1, shape)


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
# Linear regression


# CNN for CIFAR-10
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
# CNN for CIFAR-10


# AE for MNIST
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
# AE for MNIST


# Uniform Glorot initializer
def weights_init(m):
    if isinstance(m, nn.Linear):
        fan_in, fan_out = nn.init._calculate_fan_in_and_fan_out(m.weight)
        bound = math.sqrt(6 / (fan_in + fan_out))
        nn.init.uniform_(m.bias, -bound, bound)
        nn.init.uniform_(m.weight, -bound, bound)


def to_img(x):
    x = x.clamp(0, 1)
    x = x.view(x.size(0), 1, 28, 28)
    return x


network_to_use = 'AE_MNIST'  # AE_MNIST, CNN_CIFAR, LIN_REG_MNIST, ResNet_18_CIFAR
# Changing activation works only for the AE
activation = 'softplus'  # swish, softplus, relu

transforms_dict = {
        'AE_MNIST': transforms.Compose([
                           transforms.ToTensor(),
                       ]),
        'LIN_REG_MNIST': transforms.Compose([
                           transforms.ToTensor(),
                       ]),
        'CNN_CIFAR': transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]),
        'ResNet_18_CIFAR': transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    }

models = {'AE_MNIST': AE_MNIST(activation),
          'CNN_CIFAR': CIFAR_Net(),
          'LIN_REG_MNIST': LinearRegression(),
          'ResNet_18_CIFAR': ResNet18()}

criteria = {'AE_MNIST': nn.MSELoss(reduction='mean'),
            'CNN_CIFAR': nn.CrossEntropyLoss(),
            'LIN_REG_MNIST': nn.MSELoss(reduction='mean'),
            'ResNet_18_CIFAR': nn.CrossEntropyLoss()}


# Loading the data
def dataset(network_to_use_):
    if 'MNIST' in network_to_use_:
        return MNIST
    elif 'CIFAR' in network_to_use_:
        return CIFAR10

#torch.manual_seed(7)

dataset_ = dataset(network_to_use)('../data', train=True, download=True,
                                   transform=transforms_dict[network_to_use])
n = len(dataset_)
train_size = int(n*(5/6))
train_set, val_set = torch.utils.data.random_split(dataset_, [train_size, n-train_size])

test = dataset(network_to_use)('../data', train=False, download=True,
                               transform=transforms_dict[network_to_use], )

# Choosing the loss
loss_fn = criteria[network_to_use]

# Choosing the device
if torch.cuda.is_available():
    dev = 'cuda'
else:
    dev = 'cpu'

print('Using dev', dev)

# Path of the checkpoint. If there is none - set to None
start_model_path = None

# Choosing the model
model = models[network_to_use].to(dev)

#torch.manual_seed(7)

# If checkpoint is provided - load model. Otherwise - initialize the weights
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
           verbose=True,
           beta_lipschitz=1,
           eta=0.3,
           sample_size_hessian=100,
           sample_size_gradient=100
           )


# Initialize the SRC optimizer
optimizer = pytorch_optmizers.SRC(model.parameters(), opt=opt)


# Setting the device for the SRC optimizer
optimizer.defaults['dev'] = dev

sampling_scheme = dict(fixed_grad=int(optimizer.defaults['sample_size_gradient']),

                       #exponential_grad=lambda iter_: int(
                       #    min(n, n * optimizer.defaults['sample_size_gradient'] +
                       #        exp_growth_constant_grad**(iter_ + 1))),
                       fixed_hess=int(optimizer.defaults['sample_size_hessian']),
                       #exponential_hess=lambda iter_: int(
                       #    min(n, n * optimizer.defaults['sample_size_hessian'] +
                       #        exp_growth_constant_grad**(iter_ + 1))),
                       )

#torch.manual_seed(7)

def init_train_loader(dataloader_, train_, sampling_scheme_name='fixed_grad', n_points_=None):
    n_points = n_points_ if n_points_ else sampling_scheme[sampling_scheme_name]
    print('Loaded ', n_points, 'data points')
    dataloader_args = dict(shuffle=True, batch_size=n_points, num_workers=0)
    train_loader = dataloader_.DataLoader(train_, **dataloader_args)
    return dataloader_args, train_loader

