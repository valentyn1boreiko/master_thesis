from __future__ import print_function
import torch
import torch.optim as optim
from torch.autograd import Variable
from torch.optim.lr_scheduler import StepLR
import datetime
import pandas as pd
import os
import matplotlib
from config import transforms_dict, models, criteria, weights_init, dataset, parser_benchmarks

matplotlib.use('Agg')

#torch.manual_seed(7)
torch.set_printoptions(precision=10)


def train(args, model, device, train_loader, optimizer, epoch,
          test_loader, criterion, network_to_use, y_onehot):
    model.train()

    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        optimizer.samples_seen[-1] += len(data)

        if network_to_use == 'AE_MNIST':
            data = Variable(data.view(data.size(0), -1))
            target = data

        output = model(data)
        if network_to_use == 'LIN_REG_MNIST':
            y_onehot.zero_()
            y_onehot.scatter_(1, target.view(-1, 1), 1)

        loss = criterion(output, y_onehot if network_to_use == 'LIN_REG_MNIST' else target)
        loss.backward()
        optimizer.step()

        if batch_idx % args.log_interval == args.log_interval - 1 and batch_idx != 0:
            print('batch idx', batch_idx)
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       10. * batch_idx / len(train_loader), loss.item()))
            test(model, device, test_loader, train_loader, optimizer, args.log_interval, criterion,
                 network_to_use, y_onehot)

            if args.save_model:
                torch.save(model.state_dict(), "models/benchmarks_" + network_to_use + "_"
                           + args.optimizer + "_" + str(epoch) + ".pt")


def test(model, device, test_loaders, train_loader, optimizer,
         log_interval, criterion, network_to_use, y_onehot):

    model.eval()
    test_loss, train_loss = [0, 0], 0
    correct, train_correct = 0, 0
    is_classification = any(map(network_to_use.__contains__,
                                ['CNN', 'LIN_REG', 'CIFAR']))
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
                if is_classification:
                    if 'MNIST' in network_to_use:
                        y_onehot.zero_()
                        y_onehot.scatter_(1, target.view(-1, 1), 1)
                        test_loss[l_i] += criterion(output, y_onehot if network_to_use == 'LIN_REG_MNIST' else target).item() * n
                    elif 'CIFAR' in network_to_use:
                        test_loss[l_i] += criterion(output, target).item()
                elif is_AE:
                    test_loss[l_i] += criterion(output, target).item() * n
                if is_classification:
                    pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
                    correct += pred.eq(target.view_as(pred)).sum().item()

            test_loss[l_i] /= len(test_loader.dataset)

    with torch.no_grad():
        for data, target in train_loader:
            n = len(data)
            data, target = data.to(device), target.to(device)

            if network_to_use == 'AE_MNIST':
                data = Variable(data.view(data.size(0), -1))
                target = data

            output = model(data)
            if is_classification:
                if 'MNIST' in network_to_use:
                    if network_to_use == 'LIN_REG_MNIST':
                        y_onehot.zero_()
                        y_onehot.scatter_(1, target.view(-1, 1), 1)

                    train_loss += criterion(output, y_onehot if network_to_use == 'LIN_REG_MNIST' else target).item() * n
                elif 'CIFAR' in network_to_use:
                    train_loss += criterion(output, target).item()
            elif is_AE:
                train_loss += criterion(output, target).item() * n
            if is_classification:
                pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
                train_correct += pred.eq(target.view_as(pred)).sum().item()
    train_loss /= len(train_loader.dataset)


    if is_classification:
        print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            test_loss[1], correct, len(test_loader.dataset),
            100. * correct / len(test_loader.dataset)))
    elif is_AE:
        print('\nTest set: Average loss: {:.4f}\n'.format(
            test_loss[1]))


    # How many batches went through, i.e., how many computations were done

    computations_done = 0 if optimizer.first_entry \
        else optimizer.computations_done[-1] + log_interval
    
    optimizer.computations_done[-1] = computations_done
    optimizer.losses[-1] = test_loss[1]
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


def main():
    # Training settings

    args = parser_benchmarks.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    device = torch.device("cuda" if use_cuda else "cpu")

    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {'num_workers': 1}
    network_to_use = args.network_to_use
    x_axis = 'computations'  # computations, samples_seen

    model = models[network_to_use].to(device)
    model.apply(weights_init)

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


    dataset_ = dataset(network_to_use)('../data', train=True, download=True,
                                       transform=transforms_dict[network_to_use])
    n = len(dataset_)
    train_size = int(n * (5 / 6))

    train_set, val_set = torch.utils.data.random_split(dataset_, [train_size, n - train_size])

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

    print('used device - ', device)

    # For Linear Regression
    n_digits = 10
    y_onehot = torch.FloatTensor(args.batch_size, n_digits)
    # For Linear Regression

    test(model, device, test_loader, train_loader, optimizer, args.log_interval, criterion, network_to_use, y_onehot)
    for epoch in range(1, args.epochs + 1):

        train(args, model, device, train_loader, optimizer, epoch, test_loader, criterion, network_to_use, y_onehot)

        if scheduler:
            scheduler.step()


if __name__ == '__main__':
    main()
