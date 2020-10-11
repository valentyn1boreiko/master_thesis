import copy
from torch.autograd import Variable
import gc
from config import *
import autograd_hacks

parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
#torch.manual_seed(7)

def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


parser.add_argument('--subproblem-solver', type=str, default='adaptive',
                    # adaptive, non-adaptive, minres, Linear_system, Newton
                    help='subproblem solver type (default: adaptive)')
parser.add_argument('--Hessian-approx', type=str, default='AdaHess',  # AdaHess, WoodFisher, LBFGS
                    help='Hessian approximation type (default: AdaHess)')
parser.add_argument('--delta-momentum', type=str2bool, default=True,
                    help='if to use momentum for SRC (default: True)')
parser.add_argument('--AdaHess', type=str2bool, default=True,
                    help='if to use adaptive Hessian and gradient for SRC (default: True)')
parser.add_argument('--delta-momentum-stepsize', type=float, default=0.001,
                    help='momentum stepsize (default: 0.002)')
parser.add_argument('--eta', type=float, default=0.3,
                    help='learning rate for the inner subproblem (default: 0.3)')
parser.add_argument('--sigma', type=float, default=10,
                    help='Hessian Lipschitz constant (default: 10)')
parser.add_argument('--sample-size-hessian', type=int, default=10,
                    help='Hessian batch size')
parser.add_argument('--sample-size-gradient', type=int, default=100,
                    help='gradient batch size')
parser.add_argument('--epochs', type=int, default=14, metavar='N',
                    help='number of epochs to train (default: 14)')
parser.add_argument('--n-iter', type=int, default=4, metavar='N',
                    help='number of iterations of the subsolver (default: 4)')
parser.add_argument('--save-model', action='store_true', default=False,
                    help='For Saving the current Model')

args = parser.parse_args()

opt = dict(model=model,
           loss_fn=loss_fn,
           n=n,
           log_interval=1,
           problem=network_to_use,
           subproblem_solver='adaptive',  # adaptive, non-adaptive
           delta_momentum=True,
           delta_momentum_stepsize=0.01,
           initial_penalty_parameter=10,  # 15000, 10
           verbose=True,
           beta_lipschitz=1,
           eta=0.3,
           sample_size_hessian=10,
           sample_size_gradient=100
           )


def sizeof_fmt(num, suffix='B'):
    for unit in ['', 'Ki', 'Mi', 'Gi', 'Ti', 'Pi', 'Ei', 'Zi']:
        if abs(num) < 1024.0:
            return "%3.1f %s%s" % (num, unit, suffix)
        num /= 1024.0
    return "%.1f %s%s" % (num, 'Yi', suffix)


last_model = copy.deepcopy(model)


# Init train loader
dataloader_args, train_loader = init_train_loader(dataloader, train_set, n_points_=args.sample_size_gradient)
test_loader_ = dataloader.DataLoader(test, **dataloader_args)

val_loader = dataloader.DataLoader(val_set, **dataloader_args)

test_loader = (val_loader, test_loader_)

_, train_loader_hess = init_train_loader(dataloader, train_set, n_points_=args.sample_size_hessian)


dataloader_iterator = iter(train_loader)
dataloader_iterator_hess = iter(train_loader_hess)
optimizer.defaults['dataloader_iterator_hess'] = dataloader_iterator_hess
optimizer.defaults['train_loader_hess'] = train_loader_hess
optimizer.defaults['test_loader'] = test_loader
optimizer.defaults['train_loader'] = train_loader



def getBack(var_grad_fn):
    print(var_grad_fn)
    for n in var_grad_fn.next_functions:
        if n[0]:
            try:
                tensor = getattr(n[0], 'variable')
                print(n[0])
                print('Tensor with grad found:', tensor)
                print(' - gradient:', tensor.grad)
                print()
            except AttributeError as e:
                getBack(n[0])


def flatten_tensor_list(tensors):
    flattened = []
    for tensor in tensors:
        # Changed view to reshape
        flattened.append(tensor.reshape(-1))
    return torch.cat(flattened, 0)

def main():
    train_flag = True
    n_digits = 10
    if 'LIN_REG' in optimizer.defaults['problem']:
        y_onehot = torch.FloatTensor(int(optimizer.defaults['sample_size_gradient']), n_digits)
    if args.Hessian_approx == 'WoodFisher':
        autograd_hacks.add_hooks(optimizer.model)
    if optimizer.defaults['save_model']:
        print('Checkpoints will be created every 10 epochs.')
    else:
        print('Checkpoints will not be created.')

    for epoch in range(args.epochs):
        # Set modules in the the network to train mode
        print('epoch ', epoch)

        for batch_idx, (data, target) in enumerate(train_loader):
            data = data.to(optimizer.defaults['dev'])
            target = target.to(optimizer.defaults['dev'])
            if network_to_use == 'AE_MNIST':
                data = Variable(data.view(data.size(0), -1))
                target = data
            if train_flag:
                model.train()
                optimizer.model.train()
                train_flag = False

            optimizer.print_acc(len(data), epoch, batch_idx)

            optimizer.zero_grad()

            outputs = optimizer.model(data)

            if 'LIN_REG' in optimizer.defaults['problem']:
                y_onehot.zero_()
                y_onehot.scatter_(1, target.view(-1, 1).to('cpu'), 1)

            try:
                if args.Hessian_approx == 'WoodFisher':
                    autograd_hacks.clear_backprops(optimizer.model)
            except:
                pass

            loss = loss_fn(outputs, y_onehot.to(optimizer.defaults['dev']) if 'LIN_REG' in optimizer.defaults['problem'] else target)

            loss.backward(create_graph=True)

            if args.Hessian_approx == 'WoodFisher':
                autograd_hacks.compute_grad1(optimizer.model)

            optimizer.defaults['train_data'] = data
            optimizer.defaults['target'] = target

            # To get the first norm, eig
            ##optimizer.print_acc(len(data), epoch, batch_idx)

            optimizer.step()
            gc.collect()


if __name__ == '__main__':
    print('args', args)
    optimizer.defaults = update_params(optimizer.defaults, args)
    main()
