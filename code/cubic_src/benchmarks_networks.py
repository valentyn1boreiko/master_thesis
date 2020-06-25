from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torchvision import transforms
from torchvision.datasets import MNIST, CIFAR10
from torch.optim.lr_scheduler import StepLR
import matplotlib.pyplot as plt
import datetime
import pandas as pd
import os
import math
import matplotlib
matplotlib.use('Agg')


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


class AE_MNIST_torch(nn.Module):
    def __init__(self):
        super(AE_MNIST_torch, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(28 * 28, 128),
            nn.ReLU(True),
            nn.Linear(128, 64),
            nn.ReLU(True), nn.Linear(64, 12), nn.ReLU(True), nn.Linear(12, 3))
        self.decoder = nn.Sequential(
            nn.Linear(3, 12),
            nn.ReLU(True),
            nn.Linear(12, 64),
            nn.ReLU(True),
            nn.Linear(64, 128),
            nn.ReLU(True), nn.Linear(128, 28 * 28), nn.Tanh())

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


class AE_MNIST_(nn.Module):
    def __init__(self):
        super(AE_MNIST, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(28 * 28, 512),
            nn.Softplus(True),
            nn.Linear(512, 256),
            nn.Softplus(True),
            nn.Linear(256, 128),
            nn.Softplus(True),
            nn.Linear(128, 32)
        )
        self.decoder = nn.Sequential(
            nn.Linear(32, 128),
            nn.Softplus(True),
            nn.Linear(128, 256),
            nn.Softplus(True),
            nn.Linear(256, 512),
            nn.Softplus(True),
            nn.Linear(512, 28 * 28),
            nn.Tanh())

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


class AE_MNIST(nn.Module):
    def __init__(self):
        super(AE_MNIST, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(28 * 28, 512),
            nn.Softplus(threshold=float('inf')),
            nn.Linear(512, 256),
            nn.Softplus(threshold=float('inf')),
            nn.Linear(256, 128),
            nn.Softplus(threshold=float('inf')),
            nn.Linear(128, 32),
            nn.Softplus(threshold=float('inf'))
        )
        self.decoder = nn.Sequential(
            nn.Linear(32, 128),
            nn.Softplus(threshold=float('inf')),
            nn.Linear(128, 256),
            nn.Softplus(threshold=float('inf')),
            nn.Linear(256, 512),
            nn.Softplus(threshold=float('inf')),
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

# Loading the data
def dataset(network_to_use_):
    if 'MNIST' in network_to_use_:
        return MNIST
    elif 'CIFAR' in network_to_use_:
        return CIFAR10


def to_img(x):
    #x = 0.5 * (x + 1)
    x = x.clamp(0, 1)
    x = x.view(x.size(0), 1, 28, 28)
    return x


def train(args, model, device, train_loader, optimizer, epoch, test_loader, criterion, network_to_use, x_axis):
    model.train()

    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()

        if network_to_use == 'AE_MNIST':
            data = Variable(data.view(data.size(0), -1))
            target = data

        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx * len(data) % args.log_interval == 0 and batch_idx != 0:
            print('batch idx', batch_idx)
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       10. * batch_idx / len(train_loader), loss.item()))
            test(model, device, test_loader, train_loader, optimizer, batch_idx * len(data), args.log_interval, criterion,
                 network_to_use, x_axis)


def test(model, device, test_loaders, train_loader, optimizer, samples_seen_, log_interval, criterion, network_to_use, x_axis):
    model.eval()
    test_loss, train_loss = [0, 0], 0
    correct, train_correct = 0, 0
    is_CNN = 'CNN' in network_to_use
    is_AE = network_to_use == 'AE_MNIST'

    with torch.no_grad():
        for l_i, test_loader in enumerate(test_loaders):
            for data, target in test_loader:
                n = len(data)
                data, target = data.to(device), target.to(device)

                if network_to_use == 'AE_MNIST':
                    data = Variable(data.view(data.size(0), -1))
                    target = data

                output = model(data)
                if is_CNN:
                    if 'MNIST' in network_to_use:
                        test_loss[l_i] += criterion(output, target, reduction='sum').item()  # sum up batch loss
                    elif 'CIFAR' in network_to_use:
                        test_loss[l_i] += criterion(output, target).item()
                elif is_AE:
                    test_loss[l_i] += criterion(output, target).item() * n
                if is_CNN:
                    pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
                    correct += pred.eq(target.view_as(pred)).sum().item()

            test_loss[l_i] /= len(test_loader.dataset)

    if is_AE:
        _img = to_img(output)[2].reshape(28, 28)
        _img_2 = to_img(output)[3].reshape(28, 28)

        _img_target = to_img(target)[2].reshape(28, 28)
        _img_target_2 = to_img(target)[3].reshape(28, 28)

        test_img = (_img, _img_2, _img_target, _img_target_2)

    with torch.no_grad():
        for data, target in train_loader:
            n = len(data)
            data, target = data.to(device), target.to(device)

            if network_to_use == 'AE_MNIST':
                data = Variable(data.view(data.size(0), -1))
                target = data

            output = model(data)
            if is_CNN:
                if 'MNIST' in network_to_use:
                    train_loss += criterion(output, target, reduction='sum').item()  # sum up batch loss
                elif 'CIFAR' in network_to_use:
                    train_loss += criterion(output, target).item()
            elif is_AE:
                train_loss += criterion(output, target).item() * n
            if is_CNN:
                pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
                train_correct += pred.eq(target.view_as(pred)).sum().item()

    train_loss /= len(train_loader.dataset)

    if is_AE:
        _img = to_img(output)[2].reshape(28, 28)
        _img_2 = to_img(output)[3].reshape(28, 28)

        _img_target = to_img(target)[2].reshape(28, 28)
        _img_target_2 = to_img(target)[3].reshape(28, 28)


        train_img = (_img, _img_2, _img_target, _img_target_2)

    if is_CNN:
        print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            test_loss[1], correct, len(test_loader.dataset),
            100. * correct / len(test_loader.dataset)))
    elif is_AE:
        print('\nTest set: Average loss: {:.4f}\n'.format(
            test_loss[1]))

        if False and samples_seen_ % 5 * log_interval == 0:
            fig = plt.figure(figsize=(4, 8))
            for i, _img in enumerate([test_img, train_img]):
                for j in range(len(_img)):
                    fig.add_subplot(2, 2, j + 1)
                    plt.imshow(_img[j], cmap='gray')

                plt.savefig(optimizer.f_name + '_' + str(samples_seen_)
                            + '_' + ('test' if i == 0 else 'train') + '.png')
                plt.clf()

    samples_seen = 0 if optimizer.first_entry \
        else optimizer.samples_seen[-1] + log_interval

    # How many batches went through, i.e., how many computations were done
    log_interval /= n

    computations_done = 0 if optimizer.first_entry \
        else optimizer.computations_done[-1] + log_interval
    
    optimizer.computations_done[-1] = computations_done
    optimizer.samples_seen[-1] = samples_seen
    optimizer.losses[-1] = test_loss[1]
    #plt.plot(optimizer.samples_seen, optimizer.losses)
    pd.DataFrame({'samples': [optimizer.samples_seen[-1]],
                  'computations': [optimizer.computations_done[-1]],
                  'losses': [optimizer.losses[-1]],
                  'val_losses': [test_loss[0]],
                  'train_losses': [train_loss]
                  }). \
        to_csv(optimizer.f_name + '.csv',
               header=optimizer.first_entry,
               mode='w' if optimizer.first_entry else 'a',
               index=None)
    if optimizer.first_entry:
        optimizer.first_entry = False
    #plt.savefig(optimizer.f_name + '.png')


def main():
    # Training settings
    batch_size = 100
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--network-to-use', type=str, default='CNN_MNIST',  # AE_MNIST, CNN_MNIST, CNN_CIFAR
                        help='which network and problem to use (default: CNN_MNIST)')
    parser.add_argument('--optimizer', type=str, default='SGD',  # SGD, Adam, Adagrad
                        help='which optimizer to use (default: SGD)')
    parser.add_argument('--batch-size', type=int, default=batch_size, metavar='N',  # 60 for AE_MNIST, 1 for CNN_MNIST
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=batch_size, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=14, metavar='N',
                        help='number of epochs to train (default: 14)')
    parser.add_argument('--lr', type=float, default=0.001, metavar='LR',  # 0.3, 0.001, 0.005
                        help='learning rate (default: 1.0)')
    parser.add_argument('--gamma', type=float, default=0.7, metavar='M',
                        help='Learning rate step gamma (default: 0.7)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=100 * batch_size, metavar='N',
                        help='how many samples to wait before logging training status')

    parser.add_argument('--plot-interval', type=int, default=500, metavar='N',
                        help='how many samples to wait before logging training status')

    parser.add_argument('--save-model', action='store_true', default=False,
                        help='For Saving the current Model')
    parser.add_argument('--plot-results', default=True,
                        help='If to plot the results')
    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)

    device = torch.device("cuda" if use_cuda else "cpu")

    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
    network_to_use = args.network_to_use
    x_axis = 'computations'  # computations, samples_seen

    transforms_dict = {
        'CNN_MNIST': transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ]),
        'AE_MNIST': transforms.Compose([
            transforms.ToTensor(),
            #transforms.Normalize((0.5,), (0.5,))
        ]),
        'CNN_CIFAR': transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    }

    dataset_ = dataset(network_to_use)('../data', train=True, download=True,
                                       transform=transforms_dict[network_to_use])
    n = len(dataset_)
    train_size = int(n*(5/6))
    train_set, val_set = torch.utils.data.random_split(dataset_, [train_size, n-train_size])

    train_loader = torch.utils.data.DataLoader(
        train_set,
        batch_size=args.batch_size, shuffle=True, **kwargs)

    val_loader = torch.utils.data.DataLoader(
        val_set,
        batch_size=args.batch_size, shuffle=True, **kwargs)

    test_loader_ = torch.utils.data.DataLoader(
        dataset(network_to_use)('../data', train=False,
                                transform=transforms_dict[network_to_use]
                                ),
        batch_size=args.test_batch_size, shuffle=True, **kwargs)

    test_loader = (val_loader, test_loader_)
    print('Dataset sizes: ', len(list(train_loader)),
          len(list(val_loader)),
          len(list(test_loader_)))

    models = {'AE_MNIST': AE_MNIST(),  # AE_MNIST_torch, AE_MNIST
              'CNN_MNIST': Net(),
              'CNN_CIFAR': CIFAR_Net()}

    criteria = {'AE_MNIST': nn.MSELoss(reduction='mean'),
                'CNN_MNIST': F.nll_loss,
                'CNN_CIFAR': nn.CrossEntropyLoss()}

    model = models[network_to_use].to(device)
    criterion = criteria[network_to_use]

    optimizer_ = args.optimizer
    optimizers = {'SGD': optim.SGD,
                  'Adam': optim.Adam,
                  'Adagrad': optim.Adagrad}

    optimizer = optimizers[optimizer_](model.parameters(), lr=args.lr)
    print(args)
    step_size = 1
    scheduler = False
    mydir = os.path.join(os.getcwd(), 'fig',
                         'benchmarks_networks_' + datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
    try:
        os.mkdir(mydir)
    except Exception as e:
        print(str(e))

    optimizer.f_name = mydir \
                       + '/' + x_axis \
                       + '_' + network_to_use \
                       + '_' + optimizer_ \
                       + '_' + str(args.lr) + '_' \
                       + str(args.test_batch_size) + '_' \
                       + str(args.batch_size) \
                       + '_' + str(step_size) \
                       + '_scheduler=' + str(scheduler)
    print('Saving in:', optimizer.f_name)
    optimizer.samples_seen = [0]
    optimizer.losses = [0]
    optimizer.computations_done = [0]

    optimizer.first_entry = True
    if scheduler:
        scheduler = StepLR(optimizer, step_size=step_size, gamma=args.gamma)

    test(model, device, test_loader, train_loader, optimizer, 0, args.log_interval, criterion, network_to_use, x_axis)
    for epoch in range(1, args.epochs + 1):
        train(args, model, device, train_loader, optimizer, epoch, test_loader, criterion, network_to_use, x_axis)
        # test(model, device, test_loader)
        if scheduler:
            scheduler.step()

    if args.save_model:
        torch.save(model.state_dict(), "mnist_cnn.pt")


if __name__ == '__main__':
    main()
