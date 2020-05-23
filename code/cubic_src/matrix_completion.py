from torch import optim
import numpy as np
import torch
from torch.utils.data import dataloader
from torch.utils import data
from pytorch_optmizers import SRC

n_1 = 100
n_2 = 100
n_max = max(n_1, n_2)
max_iter = 2000
rank = 8
sample_rate_const = 1.
sample_rate = (n_1 + n_2) * rank * np.log(n_max) * sample_rate_const
print(sample_rate, n_1 * n_2)
print('sample_rate / n_1 * n_2 = ', sample_rate / (n_1 * n_2))
deg_o_f = rank * ((n_1 + n_2) - rank)
print('dof / sample_rate = ', deg_o_f / sample_rate)
incoherence_const = 2
print('incoherence const, n_max / rank = ', incoherence_const, n_max / rank)

A_original = torch.randn(n_1, rank)
B_original = torch.randn(n_2, rank)
C = A_original @ B_original.T
#print(C)
#print(C.symeig())
print(C.eig())
#max_eig = max(C.eig())
#min_eig = min(C.eig())
sigma_1 = 1.1206e+01
sigma_r = 7.5001e-06
conditional = sigma_1 / sigma_r

alpha_1 = np.sqrt((incoherence_const * rank * sigma_1) / n_1)
alpha_2 = np.sqrt((incoherence_const * rank * sigma_1) / n_2)

lambda_1 = n_1 / (incoherence_const * rank * conditional)
lambda_2 = n_2 / (incoherence_const * rank * conditional)

#print(C.eig())
# We assume that we know rank, otherwise - we would iterate in a loop
U = torch.randn(n_1, rank)
V = torch.randn(n_2, rank)

sampled_proportion = sample_rate / (n_1 * n_2)
not_Omega = torch.FloatTensor(C.shape).uniform_() > sampled_proportion
n = torch.sum(not_Omega == False)
print('n = ', n)


def positive_p(tensor):
    return torch.max(tensor, torch.zeros_like(tensor))


def regularized_loss(model_res, c_, u=None, v=None):
    regularizer = lambda_1 * (positive_p((u @ u.t()).diag() - alpha_1)**4).sum() \
                 + lambda_2 * (positive_p((v @ v.t()).diag() - alpha_2)**4).sum()
    return 1 / (2  * sample_rate) * ((model_res - c_).norm(p='fro'))**2\
            + regularizer
            #+ (1/8) * (u.norm(p='fro')**2 + v.norm(p='fro')**2)
            #+ (1 / 8) * ((u.t() @ u - v.t() @ v).norm(p='fro')) ** 2
            #1 / (2 * 10 * sample_rate) * ((model_res - c_).norm(p='fro'))**2
            #+ (1 / 600) * (u.norm(p='fro')**2 + v.norm(p='fro')**2)

            #+ (1 / 500) * ((u.t() @ u - v.t() @ v).norm(p='fro')) ** 2
             # + regularizer \


step_size = 0.5
momentum=0.9

eps = 1e-3

params_opt = []


UV = torch.cat((U, V))
for param_id in range(UV.shape[0]):
    temp_var = UV[param_id, :]
    temp_var.requires_grad = True
    params_opt.append(temp_var)

def index_to_params(idx, params_):
    a = torch.stack([params_[_i] for _i in idx[0]])
    b = torch.stack([params_[_i] for _i in idx[1]])
    return a, b

def model(data, index=None):
    if index:
        a, b = index_to_params(index, data)
    else:
        a, b = data[0], data[1]
    return a @ b.t()

class Dataset(data.Dataset):
    """Characterizes a dataset for PyTorch"""
    def __init__(self, c_original, not_omega_):
        """Initialization"""
        # Sample with the rate sample_rate_
        not_omega = not_omega_
        c_omega = c_original.detach().clone()
        c_omega[not_omega] = 0
        print('sampled : ', torch.sum(c_omega != 0), torch.sum(c_omega == 0))

        self.indexed_entries = [(ii, jj, c_omega[ii, jj])
                                for ii in range(c_omega.shape[0])
                                for jj in range(c_omega.shape[1])
                                if not not_omega[ii, jj]]

        print('C norm ', (c_omega - c_original).norm(p='fro'))

    def __len__(self):
        """Denotes the total number of samples"""
        return len(self.indexed_entries)

    def __getitem__(self, index):
        """Generates one sample of data"""
        # Select sample
        ii, jj, val = self.indexed_entries[index]
        u_0_ = U.shape[0]
        x = params_opt[ii], params_opt[jj + u_0_]
        # Load data and get label
        y = val

        return (ii, jj), y



#optimizer = optim.SGD(params_opt, lr=step_size, momentum=momentum)

#optimizer = optim.Adam([U, V], lr=step_size)
opt = dict(model=model,
           loss_fn=regularized_loss,
           n=n,
           log_interval=3,
           problem='matrix_completion')

if torch.cuda.is_available():
    dev = 'cuda'
else:
    dev = 'cpu'

optimizer = SRC(params_opt, opt=opt)
optimizer.defaults['dev'] = dev
optimizer.defaults['rank'] = rank
optimizer.defaults['double_sample_size'] = False

params = {'batch_size': 2,
          'shuffle': True}



sampling_scheme = dict(fixed_grad=int(n * optimizer.defaults['sample_size_gradient']),

                       #exponential_grad=lambda iter_: int(
                       #    min(n, n * optimizer.defaults['sample_size_gradient'] +
                       #        exp_growth_constant_grad**(iter_ + 1))),
                       fixed_hess=int(n * optimizer.defaults['sample_size_hessian']),
                       #exponential_hess=lambda iter_: int(
                       #    min(n, n * optimizer.defaults['sample_size_hessian'] +
                       #        exp_growth_constant_grad**(iter_ + 1))),
                       )



def init_train_loader(dataloader_, train_, sampling_scheme_name='fixed_grad', n_points_=None):
    n_points = n_points_ if n_points_ else sampling_scheme[sampling_scheme_name]
    print('Loaded ', n_points, 'data points')
    dataloader_args = dict(shuffle=True, batch_size=n_points)
    train_loader = dataloader_.DataLoader(train_, **dataloader_args)
    dataloader_args['batch_size'] = 1000
    return dataloader_args, train_loader


# Init train loader
training_set = Dataset(C, not_Omega)
dataloader_args, train_loader = init_train_loader(dataloader, training_set)

_, train_loader_hess = init_train_loader(dataloader, training_set, sampling_scheme_name='fixed_hess')

dataloader_iterator_hess = iter(train_loader_hess)


for i in range(max_iter):
    print('epoch ', i)
    for batch_idx, (data, target) in enumerate(train_loader):
        print('batch idx', batch_idx)
        #print(data, target)
        #print('u norm', params_opt[ii].norm(p='fro'))

        u_0 = U.shape[0]
        with torch.autograd.detect_anomaly():
            data_ = index_to_params(data, params_opt)
            res = model(data_)
            loss = regularized_loss(res, target, data_[0], data_[1])
            print('loss: ', loss)
            optimizer.zero_grad()
            loss.backward(create_graph=True)

        grads = []
        count = 0
        for p in optimizer.param_groups[0]['params']:
            if p.grad is not None and p.grad.sum() != 0:
                grads.append(p.grad)
                p.used = True
                p.used_id = count
                count += 1
            else:
                p.used = False

        assert count == torch.cat(data_).unique(dim=0).shape[0],\
            'grads dim is not equal to the unique number of params! '\
            + str(count) + ', ' + str(torch.cat(data_).unique(dim=0).shape[0])

        #grads = torch.autograd.grad(loss, data_, create_graph=True, only_inputs=True)


        gradsh, params = optimizer.get_grads_and_params()
        print('grads, params', gradsh.shape, gradsh.norm(p=2), len(params), torch.cat(data).shape)
        print('target len', len(target))
        optimizer.defaults['train_data'] = data
        optimizer.defaults['target'] = target
        optimizer.defaults['dataloader_iterator_hess'] = dataloader_iterator_hess
        optimizer.defaults['train_loader_hess'] = train_loader_hess

        optimizer.step()
        #print('u norm', params_opt[ii].norm(p='fro'))
        print(torch.stack(optimizer.param_groups[0]['params']).norm(p=2),
              torch.stack(optimizer.param_groups[0]['params']))

        err = (model((torch.stack(optimizer.param_groups[0]['params'][:u_0]),
                     torch.stack(optimizer.param_groups[0]['params'][u_0:]))) - C).norm(p='fro') / C.norm(p='fro')
        print('Error =', err)
        if err <= eps:
            print('Done! Iteration', i)
            break
