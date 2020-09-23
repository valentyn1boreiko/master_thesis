import copy
from torch.autograd import Variable
import gc
import psutil
import argparse
from pympler.tracker import SummaryTracker
import sys
from config import *
#torch.backends.cudnn.enabled = False

parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
"""
parser.add_argument('--sample-size-hessian', type=int, default=opt['sample_size_hessian'], metavar='N',  
                        help='input batch size for hessian (default: 300)')
parser.add_argument('--sample-size-gradient', type=int, default=opt['sample_size_gradient'], metavar='N', 
                        help='input batch size for gradient (default: 300)')
parser.add_argument('--delta-momentum-stepsize', type=float, default=opt['delta_momentum_stepsize'], metavar='LR', 
                        help='momentum stepsize (default: 0.002)')
parser.add_argument('--log-interval', type=int, default=opt['log_interval'], metavar='N',
                        help='how many samples to wait before logging training status')
"""


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
# We have T_out iterations
#tracker = SummaryTracker()

def update_params(dct, src):
    for key, value in vars(src).items():
        dct[key] = value
    return dct


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

    for epoch in range(args.epochs):
        # Set modules in the the network to train mode
        print('epoch ', epoch)

        for batch_idx, (data, target) in enumerate(train_loader):
            # ToDo: can we do so? (We increase the sample size if case 1 is not satisfied)
            # if optimizer.defaults['double_sample_size']:
            #    optimizer.defaults['double_sample_size'] = False
            #    optimizer.defaults['sample_size_gradient'] = 2 * optimizer.defaults['sample_size_gradient']
            #    optimizer.defaults['sample_size_hessian'] = 2 * optimizer.defaults['sample_size_hessian']
            #    _, train_loader = init_train_loader(dataloader, train,
            #                                        n_points_=int(optimizer.defaults['sample_size_gradient'] * optimizer.n))
            #    dataloader_iterator = iter(train_loader)
            #    _, train_loader_hess = init_train_loader(dataloader, train,
            #                                             n_points_=int(optimizer.defaults['sample_size_hessian'] * optimizer.n),
            #                                             sampling_scheme_name='fixed_hess')
            #    dataloader_iterator_hess = iter(train_loader_hess)

            # Sample g_t uniformly at random
            # try:
            #    data, target = next(dataloader_iterator)
            # except StopIteration:
            #    dataloader_iterator = iter(train_loader)
            #    data, target = next(dataloader_iterator)
            data = data.to(optimizer.defaults['dev'])
            target = target.to(optimizer.defaults['dev'])
            if network_to_use == 'AE_MNIST':
                data = Variable(data.view(data.size(0), -1))
                target = data
            if train_flag:
                model.train()
                optimizer.model.train()
                train_flag = False

            #optimizer.print_acc(len(data), epoch, batch_idx)

            #print('Memory used print_acc: ', psutil.virtual_memory().used >> 20)
            #print('Train data size ', data.size())
            optimizer.zero_grad()
            #print('Memory used zero_grad: ', psutil.virtual_memory().used >> 20)

            outputs = optimizer.model(data)

            if 'LIN_REG' in optimizer.defaults['problem']:
                y_onehot.zero_()
            if 'LIN_REG' in optimizer.defaults['problem']:
                y_onehot.scatter_(1, target.view(-1, 1), 1)

            try:
                if args.Hessian_approx == 'WoodFisher':
                    autograd_hacks.clear_backprops(optimizer.model)
            except:
                pass
            """
            for param in optimizer.param_groups[0]['params']:
                if (param.grad is None or \
                    (param.grad.sum() == 0)) \
                        and optimizer.defaults['problem'] == 'matrix_completion':
                    continue
                print(param.requires_grad)
                print('nname', param.names)
                print('each param', param.size(), param)
                print('its mean', param.grad)

            print(optimizer.model.training)
            print(model.training)
            names = [n for n, p in optimizer.model.named_parameters() if p.requires_grad]
            print(names)
            names = [n for n, p in model.named_parameters() if p.requires_grad]
            print(names)
            #exit(0)
            """
            loss = loss_fn(outputs, y_onehot if 'LIN_REG' in optimizer.defaults['problem'] else target)
            #print(getBack(loss.grad_fn))
            #grads = torch.autograd.grad(loss
            #                            , inputs=data, retain_graph=True, create_graph=True)
            #print(grads)
            #exit(0)
            loss.backward(create_graph=True)

            if args.Hessian_approx == 'WoodFisher':
                autograd_hacks.compute_grad1(optimizer.model)
            #print(getBack(loss.grad_fn))

            """
            for param in optimizer.param_groups[0]['params']:
                if (param.grad is None or \
                    (param.grad.sum() == 0)) \
                        and optimizer.defaults['problem'] == 'matrix_completion':
                    continue

                print('each param', param.size(), param)
                print('after its mean', param.grad)
            print(optimizer.get_grads_and_params()[0])
            exit(0)"""

            #loss_fn(outputs, target).backward(create_graph=True)
            #print('Memory used loss: ', psutil.virtual_memory().used >> 20)
            optimizer.defaults['train_data'] = data
            optimizer.defaults['target'] = target

            # To get the first norm, eig
            #print('Print acc!')
            optimizer.print_acc(len(data), epoch, batch_idx)

            optimizer.step()
            #print('Memory used step: ', psutil.virtual_memory().used >> 20)
            gc.collect()
            #print('Memory used collect: ', psutil.virtual_memory().used >> 20)
            #local_vars = list(locals().items())
            #for var, obj in local_vars:
            #    print(var, sys.getsizeof(obj))
            # for name, size in sorted(((name, sys.getsizeof(value)) for name, value in locals().items()),
            #                                 key= lambda x: -x[1])[:10]:
            #    print("{:>30}: {:>8}".format(name, sizeof_fmt(size)))
            # tracker.print_diff()


if __name__ == '__main__':
    print('args', args)
    optimizer.defaults = update_params(optimizer.defaults, args)
    main()
