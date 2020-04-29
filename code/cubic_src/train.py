import copy
from utils import print_acc
from config import *

last_model = copy.deepcopy(model)

dataloader_iterator = iter(train_loader)
# We have T_out iterations
for epoch in range(optimizer.defaults['n_iterations']):
    # Set modules in the the network to train mode
    model.train()

    # Sample g_t uniformly at random
    try:
        data, target = next(dataloader_iterator)
    except StopIteration:
        dataloader_iterator = iter(train_loader)
        data, target = next(dataloader_iterator)

    optimizer.zero_grad()
    loss_fn(model(data), target).backward(create_graph=True)

    optimizer.step()
    print_acc(train_loader, epoch)

    # Use sampling scheme
    _, train_loader = init_train_loader(dataloader, train)
    dataloader_iterator = iter(train_loader)

