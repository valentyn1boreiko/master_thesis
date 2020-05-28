import torch
from pytorch_optmizers import SRC
from torch import optim
import pandas as pd
import matplotlib.pyplot as plt

def model(xy):
    x, y = xy[0], xy[1]
    #return torch.sin(x) + y**2
    #return ((x**2-100)*(x**2-1))/(100/3) - 100/(100/3) + 100*y**2
    return ((x**2-0.8)*(x**2-0.01))/(100/3) - 0.008/(100/3) + 10*y**2

def convex_func(xy):
    x, y = xy[0], xy[1]
    return x**2 + y**2

opt = dict(model=model,
           loss_fn=model,
           n=1000,
           log_interval=1,
           problem='w-function',
           sample_size_hessian=0.001,
           sample_size_gradient=0.001)

problem_type = 'non-convex' # convex, non-convex
max_iter = int(3e2)
step_size = 1e-3
momentum = 0.0

optimizer_type = 'SGD' # SRC, SGD
to_plot = 'grad_norm' # grad_norm, loss

averaging_ops = 1
losses = []
grad_norms = []

samples_seen = [0]

"""
f_name = 'fig/temp/' + to_plot \
         + '_' + optimizer_type\
         + '_' + problem_type\
         + '_' + str(int(opt['sample_size_gradient'] * opt['n']))\
         + '_' + str(int(opt['sample_size_hessian'] * opt['n']))\
         + ('_' + str(step_size)) if optimizer_type == 'SGD' else '' \
         + '_' + str(averaging_ops)
"""
f_name = 'fig/temp/SGD_norm'

for op_num in range(averaging_ops):
    X = torch.tensor([-6.9655, -89.1453], requires_grad=True)
    if optimizer_type == 'SRC':
        optimizer = SRC([X], opt=opt)
    elif optimizer_type == 'SGD':
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
                            + int(opt['n'] * opt['sample_size_gradient'])
                            + int((opt['n'] * opt['sample_size_hessian'])
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

        if i == 2:
            plt.plot(samples_seen, losses if to_plot == 'loss' else grad_norms)
            plt.savefig(f_name + '.png')
        # Generate N(0, 1) perturbations of the gradient
        if optimizer_type == 'SRC':
            optimizer.perturb()

        for p in optimizer.param_groups[0]['params']:
            if optimizer_type == 'SGD':
                p.grad += (torch.randn(
                    (int(opt['n'] * opt['sample_size_gradient']),
                     p.shape[0])
                ) + p.grad).mean(dim=0)
            print('param', p, p.grad)

        optimizer.step()
        print('loss = ', loss_)

    print('Epoch {} done!'.format(op_num))

print(samples_seen)
print(losses)
pd.DataFrame({'samples': samples_seen,
              'losses'
              if to_plot == 'loss' else 'grad_norm':
                  losses if to_plot == 'loss' else grad_norms
                  }). \
         to_csv(f_name + '.csv',
               header=True,
               mode='w',
               index=None)

plt.plot(samples_seen, losses if to_plot == 'loss' else grad_norms)
plt.savefig(f_name + '.png')
