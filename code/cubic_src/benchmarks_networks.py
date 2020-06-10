from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
import matplotlib.pyplot as plt
import pandas as pd
import os


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


class AE_MNIST(nn.Module):
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


def to_img(x):
    x = 0.5 * (x + 1)
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
        if batch_idx * len(data) % args.log_interval == 0:

            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss.item()))
            test(model, device, test_loader, train_loader, optimizer, batch_idx * len(data), args.log_interval, criterion,
                 network_to_use, x_axis)


def test(model, device, test_loader, train_loader, optimizer, samples_seen_, log_interval, criterion, network_to_use, x_axis):
    model.eval()
    test_loss, train_loss = 0, 0
    correct, train_correct = 0, 0
    is_CNN = network_to_use == 'CNN_MNIST'
    is_AE = network_to_use == 'AE_MNIST'

    with torch.no_grad():
        for data, target in test_loader:
            n = len(data)
            data, target = data.to(device), target.to(device)

            if network_to_use == 'AE_MNIST':
                data = Variable(data.view(data.size(0), -1))
                target = data

            output = model(data)
            if is_CNN:
                test_loss += criterion(output, target, reduction='sum').item()  # sum up batch loss
            elif is_AE:
                test_loss += criterion(output, target).item() * n
            if is_CNN:
                pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
                correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

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
                train_loss += criterion(output, target, reduction='sum').item()  # sum up batch loss
            elif is_AE:
                train_loss += criterion(output, target).item() * n
            if is_CNN:
                pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
                train_correct += pred.eq(target.view_as(pred)).sum().item()

    train_loss /= len(train_loader.dataset)

    _img = to_img(output)[2].reshape(28, 28)
    _img_2 = to_img(output)[3].reshape(28, 28)

    _img_target = to_img(target)[2].reshape(28, 28)
    _img_target_2 = to_img(target)[3].reshape(28, 28)


    train_img = (_img, _img_2, _img_target, _img_target_2)

    if is_CNN:
        print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            test_loss, correct, len(test_loader.dataset),
            100. * correct / len(test_loader.dataset)))
    elif is_AE:
        print('\nTest set: Average loss: {:.4f}\n'.format(
            test_loss))

        if samples_seen_ / 100 % 5 == 0:
            fig = plt.figure(figsize=(4, 8))
            for i, _img in enumerate([test_img, train_img]):
                for j in range(len(_img)):
                    fig.add_subplot(2, 2, j + 1)
                    plt.imshow(_img[j], cmap='gray')

                plt.savefig(optimizer.f_name + '_' + str(samples_seen_ / 100)
                            + '_' + ('test' if i == 0 else 'train') + '.png')
                plt.clf()


    samples_seen = 0 if samples_seen_ == len(optimizer.samples_seen) == 0 \
        else optimizer.samples_seen[-1] + log_interval

    log_interval /= n

    computations_done = 0 if samples_seen_ == len(optimizer.samples_seen) == 0 \
        else optimizer.computations_done[-1] + log_interval
    
    optimizer.computations_done[-1] = computations_done
    optimizer.samples_seen[-1] = samples_seen
    optimizer.losses[-1] = test_loss
    #plt.plot(optimizer.samples_seen, optimizer.losses)
    pd.DataFrame({'samples': [optimizer.samples_seen[-1]],
                  'computations': [optimizer.computations_done[-1]],
                  'losses': [optimizer.losses[-1]],
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
    parser.add_argument('--batch-size', type=int, default=batch_size, metavar='N',  # 60 for AE_MNIST, 1 for CNN_MNIST
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=batch_size, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=14, metavar='N',
                        help='number of epochs to train (default: 14)')
    parser.add_argument('--lr', type=float, default=0.001, metavar='LR',  # 0.3, 0.001
                        help='learning rate (default: 1.0)')
    parser.add_argument('--gamma', type=float, default=0.7, metavar='M',
                        help='Learning rate step gamma (default: 0.7)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10*batch_size, metavar='N',
                        help='how many samples to wait before logging training status')

    parser.add_argument('--plot-interval', type=int, default=5, metavar='N',
                        help='how many samples to wait before logging training status')

    parser.add_argument('--save-model', action='store_true', default=False,
                        help='For Saving the current Model')
    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)

    device = torch.device("cuda" if use_cuda else "cpu")

    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
    network_to_use = 'AE_MNIST'  # AE_MNIST, CNN_MNIST
    x_axis = 'computations'  # computations, samples_seen

    transforms_dict = {
        'CNN_MNIST': transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ]),
        'AE_MNIST': transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
    }

    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=True, download=True,
                       transform=transforms_dict[network_to_use]
                       ),
        batch_size=args.batch_size, shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=False,
                       transform=transforms_dict[network_to_use]
                       ),
        batch_size=args.test_batch_size, shuffle=True, **kwargs)

    print(len(list(train_loader)), len(list(test_loader)))

    models = {'AE_MNIST': AE_MNIST(),  # AE_MNIST_torch, AE_MNIST
              'CNN_MNIST': Net()}

    criteria = {'AE_MNIST': nn.MSELoss(reduction='mean'),
                'CNN_MNIST': F.nll_loss}

    model = models[network_to_use].to(device)
    criterion = criteria[network_to_use]

    optimizer_ = 'Adam'
    optimizers = {'SGD': optim.SGD,
                  'Adam': optim.Adam,
                  'Adagrad': optim.Adagrad}

    optimizer = optimizers[optimizer_](model.parameters(), lr=args.lr)
    print(args)
    step_size = 1
    optimizer.f_name = 'fig/' \
                       + '_' + x_axis \
                       + '_' + network_to_use \
                       + '_' + optimizer_ \
                       + '_' + str(args.lr) + '_' \
                       + str(args.test_batch_size) + '_' \
                       + str(args.batch_size) + '_' + str(step_size)
    print('Saving in:', optimizer.f_name)
    optimizer.samples_seen = [0]
    optimizer.losses = [0]
    optimizer.computations_done = [0]

    optimizer.first_entry = True
    scheduler = StepLR(optimizer, step_size=step_size, gamma=args.gamma)

    test(model, device, test_loader, train_loader, optimizer, 0, args.log_interval, criterion, network_to_use, x_axis)
    for epoch in range(1, args.epochs + 1):
        train(args, model, device, train_loader, optimizer, epoch, test_loader, criterion, network_to_use, x_axis)
        # test(model, device, test_loader)
        scheduler.step()

    if args.save_model:
        torch.save(model.state_dict(), "mnist_cnn.pt")


if __name__ == '__main__':
    main()
