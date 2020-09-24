import torch
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
import pandas as pd

torch.manual_seed(7)
torch.set_printoptions(precision=10)


transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
"""
trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=100,
                                          shuffle=True, num_workers=1)
"""
dataset_ = torchvision.datasets.CIFAR10('./data', train=True, download=True,
                                       transform=transform)
n = len(dataset_)
train_size = int(n * (5 / 6))

train_set, val_set = torch.utils.data.random_split(dataset_, [train_size, n - train_size])

trainloader = torch.utils.data.DataLoader(
    train_set,
    batch_size=100, shuffle=True, num_workers=1)


val_loader = torch.utils.data.DataLoader(
    val_set,
    batch_size=100, shuffle=True, num_workers=1)

testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=100,
                                         shuffle=False, num_workers=1)

import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
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

device = 'cpu'

net = Net().to(device)

criterion = nn.CrossEntropyLoss()
lr = 0.1
optimizer = optim.SGD(net.parameters(), lr)

samples = 0
first_entry = True
samples_seen = 0
f_name = 'SGD_lr=' + str(lr)

for epoch in range(14):  # loop over the dataset multiple times

    running_loss = 0.0
    if first_entry:
        net.eval()
        with torch.no_grad():
            train_loss = 0
            for data, target in trainloader:
                n = len(data)
                data, target = data.to(device), target.to(device)
                output = net(data)
                train_loss += criterion(output, target).item()

                # print(len(trainloader.dataset))

        train_loss /= len(trainloader.dataset)
        print('[%d, %5d] loss: %.10f' %
              (epoch + 1, 0, train_loss))

        pd.DataFrame({'samples': [samples_seen],
                      'train_losses': [train_loss]
                      }). \
            to_csv(f_name + '.csv',
                   header=first_entry,
                   mode='w' if first_entry else 'a',
                   index=None)

        first_entry = False


    for i, data in enumerate(trainloader, 0):
        net.train()
        print(len(trainloader))

        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        samples_seen += 100

        # print statistics
        #running_loss += loss.item()
        #if i % 2000 == 1999:    # print every 2000 mini-batches

        if i % 100 == 99:
            net.eval()
            with torch.no_grad():
                train_loss = 0
                for data, target in trainloader:
                    n = len(data)
                    data, target = data.to(device), target.to(device)
                    output = net(data)
                    train_loss += criterion(output, target).item()

                    # print(len(trainloader.dataset))

            train_loss /= len(trainloader.dataset)
            print('[%d, %5d] loss: %.10f' %
                  (epoch + 1, 0, train_loss))

            pd.DataFrame({'samples': [samples_seen],
                          'train_losses': [train_loss]
                          }). \
                to_csv(f_name + '.csv',
                       header=first_entry,
                       mode='w' if first_entry else 'a',
                       index=None)

        #running_loss = 0.0

print('Finished Training')

