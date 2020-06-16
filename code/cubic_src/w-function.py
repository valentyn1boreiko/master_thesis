import torch
from pytorch_optmizers import SRC
from torch import optim
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


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

    #return torch.sin(x) + y**2
    #return ((x**2-100)*(x**2-1))/(100/3) - 100/(100/3) + 100*y**2
    #return ((x**2-0.8)*(x**2-0.01))/(100/3) - 0.008/(100/3) + 10*y**2
    return w + 10*y**2


def convex_func(xy):
    x, y = xy[0], xy[1]
    return x**2 + y**2


opt = dict(model=model,
           loss_fn=model,
           n=1000,
           log_interval=100,
           problem='w-function',
           sample_size_hessian=300,  # 0.3
           sample_size_gradient=300,  # 0.3
           subproblem_solver='non-adaptive',
           eta=0.3,
           n_iter=5,
           initial_penalty_parameter=2,  # 150, 1500
           beta_lipschitz=1,
           delta_momentum=False,
           delta_momentum_stepsize=.9)  # 0.9

problem_type = 'non-convex'  # convex, non-convex
max_iter = int(4e2)
step_size = 0.04  # 0.04, 0.001
momentum = 0.9  # 0.7

optimizer_type = 'SRC'  # SRC, SGD, Adam
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
    X = torch.tensor([10.0, 10.0], requires_grad=True)
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
            # Nr of gradient samples
            samples_seen.append(samples_seen[-1]
                            + int(opt['sample_size_gradient'])
                            + int((opt['sample_size_hessian'])
                                  if optimizer_type == 'SRC' else 0)
                            )

        print('idx ', i, losses
        if to_plot == 'loss' else grad_norms,
              samples_seen)

        optimizer.zero_grad()
        loss_.backward(create_graph=True)
        if to_plot == 'grad_norm':
            for p in optimizer.param_groups[0]['params']:
                grad_norms.append(p.norm(p=2))
        """
        if i == 2:
            plt.plot(optimizer.computations_done[:-1] if optimizer_type == 'SRC' else samples_seen,
                     losses if to_plot == 'loss' else grad_norms)
            plt.savefig(f_name + '.png')
        """
        # Generate N(0, 1) perturbations of the gradient
        if optimizer_type == 'SRC':
            optimizer.perturb()

        for p in optimizer.param_groups[0]['params']:
            if optimizer_type == 'SGD':
                p.grad += (torch.randn(
                    (int(opt['sample_size_gradient']),
                     p.shape[0])
                ) + p.grad).mean(dim=0)
            print('param', p, p.grad)

        optimizer.step()
        print('loss = ', loss_.item())
        if optimizer_type == 'SRC':
            computations_done_src.append(optimizer.computations_done)
            grad_norms_src.append(optimizer.grad_norms)
            least_eig_src.append(optimizer.least_eig)
            print(losses, computations_done_src)
            #assert len(losses) == len(optimizer.computations_done[:-1]), 'losses != computations_done !'

    print('Epoch {} done!'.format(op_num))

print(samples_seen)
print(losses)
pd.DataFrame({'samples': samples_seen,
              'computations': list(np.array(samples_seen) / int(opt['sample_size_gradient']))
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
    plt.plot(list(np.array(samples_seen) / int(opt['sample_size_gradient'])),
             losses if to_plot == 'loss' else grad_norms, label='loss')
elif optimizer_type == 'SRC':
    plt.plot(computations_done_src[120:-1], losses[120:] if to_plot == 'loss' else grad_norms, label='loss')
plt.legend()
plt.savefig(f_name + '.png')
plt.clf()
if optimizer_type == 'SRC':
    plt.plot(computations_done_src[:-1], grad_norms_src, label='grad_norms')
    plt.plot(computations_done_src[:-1], least_eig_src, label='least_eig')
    plt.legend()
    plt.savefig(f_name + '_eig_norms_' + '.png')

print('Saving in:', f_name)
