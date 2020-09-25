import torch
from pytorch_optmizers import SRC
from torch import optim
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from config import wfunction_parser, update_params

torch.manual_seed(7)
def model(xy):
    eps = 0.01
    L = 5

    x, y = xy[0], xy[1]
    if x <= -L * np.sqrt(eps):
        w = np.sqrt(eps) * (x + (L + 1)*np.sqrt(eps))**2 \
            - 1/3 * (x + (L + 1)*np.sqrt(eps))**3 \
            - 1/3 * (3*L + 1)*eps**(3/2)
    elif -L * np.sqrt(eps) < x <= - np.sqrt(eps):
        w = eps * x + eps**(3/2) / 3
    elif -np.sqrt(eps) < x <= 0:
        w = -np.sqrt(eps) * x**2 - x**3 / 3
    elif 0 < x <= np.sqrt(eps):
        w = -np.sqrt(eps) * x**2 + x**3 / 3
    elif np.sqrt(eps) < x <= L*np.sqrt(eps):
        w = -eps * x + eps**(3/2) / 3
    elif x >= L * np.sqrt(eps):
        w = np.sqrt(eps) * (x - (L + 1) * np.sqrt(eps)) ** 2 \
            + 1 / 3 * (x - (L + 1) * np.sqrt(eps)) ** 3 \
            - 1 / 3 * (3 * L + 1) * eps ** (3 / 2)

    return w + 10*y**2


def convex_func(xy):
    x, y = xy[0], xy[1]
    return x**2 + y**2


args = wfunction_parser.parse_args()

opt = dict(model=model,
           loss_fn=model,
           n=1000,
           log_interval=100,
           problem='w-function',
           sample_size_hessian=300,
           sample_size_gradient=300,
           subproblem_solver='non-adaptive',
           Hessian_approx='None',
           eta=0.01,  # 0.05
           n_iter=10,
           initial_penalty_parameter=1,
           beta_lipschitz=1,
           delta_momentum=False,
           delta_momentum_stepsize=.9,
           verbose=True)

opt = update_params(opt, args)

problem_type = 'non-convex'  # convex, non-convex
max_iter = int(8e2)  # 1e4 for SRC, 8e2 for SGD
step_size = opt['eta']  # 0.04, 0.001
momentum = 0.0  # 0.7

optimizer_type = args.optimizer  # SRC, SGD, Adam
to_plot = 'loss'  # grad_norm, loss

averaging_ops = 1
losses = []
grad_norms = []

samples_seen = [0]
computations_done_src = [0]
grad_norms_src = []
least_eig_src = []

if optimizer_type == 'SGD':
    f_name = 'fig/w-function/computations_momentum_' + to_plot \
         + '_' + optimizer_type\
         + '_' + problem_type\
         + '_' + str(int(opt['sample_size_gradient']))\
         + '_' + str(int(opt['sample_size_hessian']))\
         + '_' + str(step_size) \
         + '_' + str(momentum) \
         + '_' + str(averaging_ops)

elif optimizer_type == 'SRC':
    f_name = 'fig/w-function/computations_' + to_plot \
             + '_' + optimizer_type \
             + '_' + problem_type \
             + '_' + str(int(opt['sample_size_gradient'])) \
             + '_' + str(int(opt['sample_size_hessian'])) \
             + '_' + opt['subproblem_solver'] \
             + '_' + str(opt['eta']) \
             + '_' + str(opt['n_iter']) \
             + '_' + str(opt['initial_penalty_parameter']) \
             + '_' + str(opt['beta_lipschitz']) \
             + '_' + str(opt['delta_momentum']) \
             + '_' + str(opt['delta_momentum_stepsize']) \
             + '_' + str(averaging_ops)

elif optimizer_type == 'Adam':
    f_name = 'fig/w-function/computations' \
             + '_' + to_plot \
             + '_' + optimizer_type \
             + '_' + problem_type \
             + '_' + str(int(opt['sample_size_gradient'])) \
             + '_' + str(int(opt['sample_size_hessian'])) \
             + '_' + str(step_size) \
             + '_' + str(averaging_ops)

for op_num in range(averaging_ops):
    X = torch.tensor([0.0, 0.0], requires_grad=True)
    if optimizer_type == 'SRC':
        optimizer = SRC([X], opt=opt)
    elif optimizer_type == 'SGD':
        optimizer = optim.SGD([X], lr=step_size, momentum=momentum)
    elif optimizer_type == 'Adam':
        optimizer = optim.SGD([X], lr=step_size, momentum=momentum)

    for i in range(max_iter):
        loss_ = model(X)
        if to_plot == 'loss':
            if op_num == 0:
                losses.append(loss_.item())
            else:
                # Running mean
                if i > 0:
                    losses[i] = (losses[i] * op_num + loss_.item()) / (op_num + 1)

        if op_num == 0 and i > 0:
            samples_seen.append(samples_seen[-1] + int(opt['sample_size_gradient']))

        optimizer.zero_grad()
        loss_.backward(create_graph=True)
        if to_plot == 'grad_norm':
            for p in optimizer.param_groups[0]['params']:
                grad_norms.append(p.norm(p=2))

        # Generate N(0, 1) perturbations of the gradient
        if optimizer_type == 'SRC':
            optimizer.perturb()

        for p in optimizer.param_groups[0]['params']:
            if optimizer_type == 'SGD':
                p.grad += (torch.randn(
                    (int(opt['sample_size_gradient']),
                     p.shape[0])
                ) + p.grad).mean(dim=0)

        optimizer.step()
        if optimizer_type == 'SRC':
            computations_done_src.append(optimizer.computations_done_times_samples)
            grad_norms_src.append(optimizer.grad_norms)
            least_eig_src.append(optimizer.least_eig)

    print('Epoch {} done!'.format(op_num))

pd.DataFrame({'computations_done_times_samples': samples_seen
              if optimizer_type in ['Adam', 'SGD'] else computations_done_src[:-1],
              'losses'
              if to_plot == 'loss' else 'grad_norm':
                  losses if to_plot == 'loss' else grad_norms
                  }). \
         to_csv(f_name + '.csv',
               header=True,
               mode='w',
               index=None)

if optimizer_type in ['Adam', 'SGD']:
    plt.plot(samples_seen,
             losses if to_plot == 'loss' else grad_norms, label='loss')
elif optimizer_type == 'SRC':
    plt.plot(computations_done_src[:-1], losses[:] if to_plot == 'loss' else grad_norms, label='loss')
plt.legend()
plt.savefig(f_name + '.png')
plt.clf()

if optimizer_type == 'SRC':
    plt.plot(computations_done_src[:-1], grad_norms_src)
    plt.xlabel('Oracle calls')
    plt.ylabel('Gradient norm')
    plt.title('W-function')
    plt.legend()
    plt.savefig(f_name + '_eig_norms_' + '.png')

print('Saving in:', f_name)
