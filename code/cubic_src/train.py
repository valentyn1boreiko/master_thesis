import copy
from config import *

last_model = copy.deepcopy(model)

dataloader_iterator = iter(train_loader)
dataloader_iterator_hess = iter(train_loader_hess)
# We have T_out iterations
for epoch in range(optimizer.defaults['n_iterations']):
    # Set modules in the the network to train mode
    print('epoch ', epoch)
    model.train()

    # ToDo: can we do so? (We increase the sample size if case 1 is not satisfied)
    if optimizer.defaults['double_sample_size']:
        optimizer.defaults['double_sample_size'] = False
        optimizer.defaults['sample_size_gradient'] *= 2
        optimizer.defaults['sample_size_hessian'] *= 2
        _, train_loader = init_train_loader(dataloader, train)
        dataloader_iterator = iter(train_loader)

    # Sample g_t uniformly at random
    try:
        data, target = next(dataloader_iterator)
    except StopIteration:
        dataloader_iterator = iter(train_loader)
        data, target = next(dataloader_iterator)

    optimizer.zero_grad()
    loss_fn(model(data), target).backward(create_graph=True)
    optimizer.defaults['train_data'] = data
    optimizer.defaults['target'] = target
    optimizer.defaults['dataloader_iterator_hess'] = dataloader_iterator_hess

    optimizer.step()
    optimizer.print_acc(train_loader, epoch)

    # Use sampling scheme
    _, train_loader = init_train_loader(dataloader, train)
    dataloader_iterator = iter(train_loader)

