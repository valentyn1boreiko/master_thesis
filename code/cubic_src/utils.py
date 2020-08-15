import builtins
import json
import logging

import torch
import time
import numpy as np
import pandas as pd
import scipy.linalg as la
from scipy.sparse.linalg import minres
from torch.autograd import Variable
from torch.optim.optimizer import Optimizer
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os, datetime
from scipy.sparse.linalg import LinearOperator
# Comment it out while using matrix_completion.py or w-function.py instead of train.py
import config

temp_dot = torch.dot
torch.dot = None
torch.dot = lambda x, y: temp_dot(x.to(torch.float32), y.to(torch.float32))

def lanczos_tridiag_to_diag(t_mat):
    """
    Given a num_init_vecs x num_batch x k x k tridiagonal matrix t_mat,
    returns a num_init_vecs x num_batch x k set of eigenvalues
    and a num_init_vecs x num_batch x k x k set of eigenvectors.

    TODO: make the eigenvalue computations done in batch mode.
    """
    orig_device = t_mat.device
    if t_mat.size(-1) < 32:
        retr = torch.symeig(t_mat.cpu(), eigenvectors=True)
    else:
        retr = torch.symeig(t_mat, eigenvectors=True)

    evals, evecs = retr
    mask = evals.ge(0)
    evecs = evecs * mask.type_as(evecs).unsqueeze(-2)
    evals = evals.masked_fill_(~mask, 1)

    return evals.to(orig_device), evecs.to(orig_device) 

def init_train_loader(dataloader, train, sampling_scheme_name='fixed'):
    dataloader_args = dict(shuffle=True, batch_size=config.sampling_scheme[sampling_scheme_name], num_workers=4)
    train_loader = dataloader.DataLoader(train, **dataloader_args)
    return dataloader_args, train_loader


def flatten_tensor_list(tensors):
    flattened = []
    for tensor in tensors:
        # Changed view to reshape
        flattened.append(tensor.reshape(-1))
    return torch.cat(flattened, 0)


def unflatten_tensor_list(tensor, params):
    unflattened = []
    step = 0
    for param in params:
        n = param.reshape(-1).size()[0]
        unflattened.append(tensor[step: step + n].reshape(param.size()))
        step += n
    return unflattened


def tridiag(a, b, c, k1=-1, k2=0, k3=1):
    return torch.diag(a, k1) + torch.diag(b, k2) + torch.diag(c, k3)


def sample_spherical(npoints, ndim=3):
    vec = torch.randn(ndim, npoints)
    vec /= vec.norm(p=2, dim=0)
    return vec


def index_to_params(idx, params_):
    a = torch.stack([params_[_i] for _i in idx[0]])
    b = torch.stack([params_[_i] for _i in idx[1]])
    return a, b


def verbose_decorator_print(verbose):
    '''filename is the file where output will be written'''
    def wrap(func):
        '''func is the function you are "overriding", i.e. wrapping'''
        def wrapped_func(*args, **kwargs):
            '''*args and **kwargs are the arguments supplied
            to the overridden function'''
            #use with statement to open, write to, and close the file safely

            #now original function executed with its arguments as normal
            if verbose:
                return func(*args, **kwargs)
            else:
                pass
        return wrapped_func
    return wrap


class SRCutils(Optimizer):
    def __init__(self, params,
                 batchsize_mode='fixed', opt=None):

        if opt is None:
            opt = dict()

        self.model = opt['model']
        self.loss_fn = opt['loss_fn']
        self.case_n = 1
        self.n = opt['n']
        self.grad, self.params = None, None
        self.samples_seen = 0
        self.computations_done = 0
        self.computations_done_times_samples = 0
        self.test_losses = 0
        self.running_update = 0
        self.mean_update = 0
        self.least_eig = None
        self.grad_norms = None
        self.step_old = None
        self.delta_m_norm = None

        self.first_hv = True

        # Momentum, experimental
        self.b_1 = 0.9
        self.b_2 = 0.999
        self.m = 0
        self.v = 0
        self.t = 0
        self.t_grad = 0
        self.t_hess = 0

        self.g_ = 0
        self.D = 0
        self.D_hat = None
        self.epsilon = 1e-08

        self.defaults = dict(problem=opt.get('problem', 'CNN'),  #matrix_completion, CNN, w-function, AE
                             grad_tol=opt.get('grad_tol', 1e-2),
                             activation=opt.get('activation', 'swish'),
                             adaptive_rho=opt.get('adaptive_rho', False),
                             subproblem_solver=opt.get('subproblem_solver', 'adaptive'),
                             batchsize_mode=batchsize_mode,
                             sample_size_hessian=opt.get('sample_size_hessian', 0.03 / 6),
                             sample_size_gradient=opt.get('sample_size_gradient', 0.03 / 6),
                             eta_1=opt.get('success_treshold', 0.02),
                             eta_2=opt.get('very_success_treshold', 0.1),
                             gamma=opt.get('penalty_increase_decrease_multiplier', 2),  # 2
                             sigma=opt.get('initial_penalty_parameter', 0.1),  # 16
                             beta_lipschitz=opt.get('beta_lipschitz', None),
                             eta=opt.get('eta', None),
                             n_epochs=opt.get('n_epochs', 14),
                             target=None,
                             log_interval=opt.get('log_interval', 600),
                             plot_interval=opt.get('plot_interval', 5),
                             AdaHess=opt.get('AdaHess', False),
                             delta_momentum=opt.get('delta_momentum', False),
                             delta_momentum_stepsize=opt.get('delta_momentum_stepsize', 0.05),  # 0.04
                             AccGD=opt.get('AccGD', False),
                             innerAdam=opt.get('innerAdam', False),
                             verbose=opt.get('verbose', False),
                             n_iter=opt.get('n_iter', 4),
                             block_size=opt.get('block_size', None),
                             momentum_schedule_linear_const=opt.get('schedule_linear', 1.0),  # scale the momentum step-size
                             momentum_schedule_linear_period=opt.get('schedule_linear_period', 10)
                             )

        builtins.print = verbose_decorator_print(self.defaults['verbose'])(print)
        self.first_entry = True
        self.is_matrix_completion = self.defaults['problem'] == 'matrix_completion'
        self.is_w_function = self.defaults['problem'] == 'w-function'
        self.is_classification = any(map(self.defaults['problem'].__contains__,
                                         ['CNN', 'LIN_REG']))
        self.is_AE = 'AE' in self.defaults['problem']
        self.mydir = os.path.join(os.getcwd(), 'fig', datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
        try:
            os.mkdir(self.mydir)
        except Exception as e:
            print(str(e))

        super(SRCutils, self).__init__(params, self.defaults)

    def perturb(self, type_='gradient', hv=None):
        if type_ == 'gradient':
            for p in self.param_groups[0]['params']:
                # Generate N(0, 1) perturbations of the gradient
                p.grad = (torch.randn(
                    (self.get_num_points(),
                     p.shape[0])
                ) + p.grad).mean(dim=0)

        elif type_ == 'hessian':
            hv = (torch.randn(
                    (self.get_num_points(type_=type_),
                     hv.shape[0])
                ) + hv).mean(dim=0)

            return hv

    def get_num_points(self, type_='gradient'):
        return int(self.defaults['sample_size_' + type_])

    def get_accuracy(self, loaders):
        #print(1)
        self.model.eval()
        #print(2)
        nn = len(loaders)
        acc, correct, total, loss = ([0]*nn, [0]*nn, [0]*nn, [0]*nn)
        torch.manual_seed(7)

        with torch.no_grad():
            for l_i, loader_ in enumerate(loaders):
                torch.manual_seed(7)

                for batch_idx_, (data_, target_) in enumerate(loader_):
                    # Get Samples
                    n = len(data_)
                    if self.is_AE:
                        data_ = Variable(data_.view(data_.size(0), -1))
                        target_ = data_
                    #print(3)
                    data_ = data_.to(self.defaults['dev'])
                    target_ = target_.to(self.defaults['dev'])
                    torch.manual_seed(7)
                    #print(3, 4)
                    #print(data_.is_cuda, self.defaults['dev'], next(self.model.parameters()).is_cuda, self.model)
                    outputs = self.model(data_)
                    #print(4)
                    #print(data_.norm(p=2), outputs.norm(p=2))
                    if self.is_AE:
                        loss[l_i] += self.loss_fn(outputs, target_).item() * n
                    elif 'LIN_REG' in self.defaults['problem']:
                        #print(target_, outputs)
                        self.y_onehot.zero_()
                        self.y_onehot.scatter_(1, target_.view(-1, 1), 1)
                        self.y_onehot.scatter_(1, target_.view(-1, 1), 1)
                        #print('Outputs!', outputs[:3], self.y_onehot[:3])

                        #print(outputs.size(), self.y_onehot.size())
                        loss[l_i] += self.loss_fn(outputs, self.y_onehot).item() * n
                        #loss[l_i] += self.loss_fn(outputs, target_).item() * n # reduction = 'sum'
                        #print(4)
                    elif 'CIFAR' in self.defaults['problem']:
                        loss[l_i] += self.loss_fn(outputs, target_).item()
                    # Get prediction
                    _, predicted = torch.max(outputs.data, 1)
                    #print(5)
                    # Total number of labels
                    total[l_i] += len(target_)
                    # Total correct predictions
                    if not self.is_AE:
                        correct[l_i] += (predicted == target_).sum().detach()
                    del predicted
                if self.is_AE:
                    img_ = config.to_img(outputs)[2].reshape(28, 28)
                    img_2 = config.to_img(outputs)[3].reshape(28, 28)

                    img_target = config.to_img(target_)[2].reshape(28, 28)
                    img_target_2 = config.to_img(target_)[3].reshape(28, 28)
                del outputs, target_

                loss[l_i] = loss[l_i] / len(loader_.dataset)
                #print(loss[l_i], len(loader_.dataset), self.model.training)
                #exit(0)

                if not self.is_AE:
                    acc[l_i] = 100 * correct[l_i] / total[l_i]


        print("All points {}".format(total))

        if nn == 1:
            loss = loss[0]
            acc = acc[0]
        return loss, (acc if not self.is_AE else (img_, img_2, img_target, img_target_2))

    def print_acc(self, batch_size, epoch, batch_idx):
        if self.first_entry:
            n_digits = 10
            self.y_onehot = torch.FloatTensor(
                int(self.defaults['sample_size_gradient']), n_digits)
            self.y_onehot_hess = torch.FloatTensor(
                int(self.defaults['sample_size_hessian']), n_digits)
            self.f_name = self.mydir + '/loss_src' \
                          + '_delta=' + str(self.defaults['delta_momentum']) \
                          + '_AdaHess=' + str(self.defaults['AdaHess']) \
                          + '_sigma=' + str(self.defaults['sigma']) \
                          + '_delta_step=' + str(self.defaults['delta_momentum_stepsize']) \
                          + '_eta=' + str(self.defaults['eta']) \
                          + '_' + self.defaults['problem'] \
                          + '_' + self.defaults['subproblem_solver'] \
                          + '_H_size=' + str(self.defaults['sample_size_hessian']) \
                          + '_g_size=' + str(self.defaults['sample_size_gradient'])
            blacklist_keys = ['target', 'train_data', 'dataloader_iterator_hess',
                              'train_loader_hess', 'test_loader', 'train_loader']
            settings_dict = dict([(key, val) for key, val in self.defaults.items() if key not in blacklist_keys])
            json.dump(settings_dict, open(self.mydir + '/settings.json', 'w'))

            print('Saving in:', self.f_name)

        #train_loss, train_acc = self.get_accuracy(train_loader)
        #print(
        #    "Epoch {} Train Loss: {:.4f} Accuracy :{:.4f} Test Loss: {:.4f} Accuracy: {:.4f}".format(epoch, train_loss,
        #                                                                                             train_acc,
        #                                                                                             test_loss,
        #                                                                                             test_acc))

        if batch_idx % self.defaults['log_interval'] == 0:
            test_loss, test_acc = self.get_accuracy(self.defaults['test_loader'])
            train_loss, train_acc = self.get_accuracy([self.defaults['train_loader']])

            if self.is_AE:
                if False and batch_idx % self.defaults['plot_interval'] == 0:
                    fig = plt.figure(figsize=(8, 8))
                    for i, _img in enumerate([test_acc, train_acc]):
                        for j in range(len(_img)):
                            fig.add_subplot(2, 2, j+1)
                            plt.imshow(_img[j], cmap='gray')


                        plt.savefig(self.mydir + '/' + str(epoch) + '_' +
                                    str(batch_idx * batch_size) + ('test' if i == 0 else 'train') + '.png')
                        plt.clf()


            if not self.is_AE:
                print(
                    "Epoch {} Test Loss: {:.4f} Accuracy: {:.4f}".format(epoch,
                                                                         test_loss[1],
                                                                         test_acc[1]
                                                                         )
                )
            else:
                print(
                    "Epoch {} Test Loss: {:.4f}".format(epoch,
                                                        test_loss[1]
                                                        )
                )

            self.test_losses = test_loss[1]
            #plt.plot(self.computations_done, self.test_losses, label='loss')
            #plt.legend()
            print('batch id', batch_idx)

            pd.DataFrame({'samples': [self.samples_seen],
                          'computations': [self.computations_done],
                          'computations_times_sample': [self.computations_done_times_samples],
                          'losses': [self.test_losses],
                          'val_losses': test_loss[0],
                          'train_losses': [train_loss],
                          'grad_norms': [self.grad_norms],
                          'least_eig': [self.least_eig if self.least_eig else None],
                          'delta_m_norm': [self.delta_m_norm]
                          }).\
                to_csv(self.f_name + '.csv',
                       header=self.first_entry,
                       mode='w' if self.first_entry else 'a',
                       index=None)
            if self.first_entry:
                self.first_entry = False
            #plt.savefig(self.f_name + '.png')
            #plt.clf()
            print('idx ', batch_idx, self.test_losses, self.samples_seen)

    def cauchy_point(self):
        # Compute Cauchy radius
        # ToDo: replace hessian-vec product with the upper bound (beta)
        grads_norm = self.grad.norm(p=2)
        print('Cauchy point', grads_norm)
        product = torch.dot(self.hessian_vector_product(self.grad), self.grad) / (self.defaults['sigma'] * grads_norm ** 2)  # @
        print('product', product, product.size())
        if product <= 0:
            print('Negative product!', product)
        R_c = -product + torch.sqrt(product ** 2 + 2 * grads_norm / self.defaults['sigma'])
        print('Rc', R_c)
        delta = -R_c * self.grad / grads_norm
        print(delta.norm(p=2))
        return delta.detach()  # if product > 0 else torch.zeros(self.grad.size())

    def get_eigen(self, H_bmm, matrix=None, maxIter=5, tol=1e-3, method='lanczos', which='biggest'):
        """
        compute the top eigenvalues of model parameters and
        the corresponding eigenvectors.
        """
        # change the model to evaluation mode, otherwise the batch Normalization Layer will change.
        # If you call this function during training, remember to change the mode back to training mode.
        _, params = self.get_grads_and_params()

        if params:
            q = flatten_tensor_list([torch.randn(p.size(), device=p.device) for p in params])
        else:
            q = torch.randn(matrix.size()[0])
        q = q / torch.norm(q)

        eigenvalue = None

        if method == 'power' and which != 'least':
            # Power iteration
            for _ in range(maxIter):
                self.computations_done += 1
                self.computations_done_times_samples += self.defaults['sample_size_gradient']
                Hv = H_bmm(q)
                eigenvalue_tmp = torch.dot(Hv, q)
                Hv_norm = torch.norm(Hv)
                if Hv_norm == 0:
                    break
                q = Hv / Hv_norm
                if eigenvalue is None:
                    eigenvalue = eigenvalue_tmp
                else:
                    if abs(eigenvalue - eigenvalue_tmp) / abs(eigenvalue) < tol:
                        return eigenvalue_tmp.item()
                    else:
                        eigenvalue = eigenvalue_tmp
            return eigenvalue

        elif method == 'lanczos' or which == 'least':
            # Lanczos iteration
            b = 0
            if params:
                q_last = flatten_tensor_list([torch.zeros(p.size(), device=p.device) for p in params])
            else:
                q_last = torch.zeros(matrix.size()[0])
            #q_s = [q_last]
            a_s = []
            b_s = []
            for _ in range(maxIter):
                if which == 'biggest':
                    self.computations_done += 1
                    self.computations_done_times_samples += self.defaults['sample_size_gradient']
                print('before', q.dtype)
                Hv = H_bmm(q)
                print(Hv.dtype, q.dtype)
                a = torch.dot(Hv, q)
                Hv -= (b * q_last + a * q)
                q_last = q
                #q_s.append(q_last)
                b = torch.norm(Hv)
                a_s.append(a)
                b_s.append(b)
                if b == 0:
                    break
                q = Hv / b
            eigs, _ = la.eigh_tridiagonal(a_s, b_s[:-1])
            #a_s = torch.tensor(a_s).to(self.defaults['dev'])
            #b_s = torch.tensor(b_s[:-1]).to(self.defaults['dev'])
            #eigs, _ = lanczos_tridiag_to_diag(torch.diag_embed(a_s) + torch.diag_embed(b_s, offset=-1) + torch.diag_embed(b_s, offset=1))
            return max(abs(eigs)) if which == 'biggest' else min(eigs)

    def get_hessian_eigen(self, **kwargs):
        H_bmm = lambda x: self.hessian_vector_product(x)
        return self.get_eigen(H_bmm, **kwargs)

    def get_grads_and_params(self):
        """ Get model parameters and corresponding gradients
        """
        params = []
        grads = []
        # We assume only one group for now
        for param in self.param_groups[0]['params']:
            if (not (param.grad is None) and not (param.grad.sum() == 0)) \
                    or self.defaults['problem'] != 'matrix_completion':
                params.append(param)
            if param.grad is None or \
                    (param.grad.sum() == 0
                     and self.defaults['problem'] == 'matrix_completion'):
                continue

            grads.append(param.grad + 0.)
        return flatten_tensor_list(grads), params

    def update_params(self, delta, inplace=True):
        param_deltas = unflatten_tensor_list(delta, self.params)
        assert (len(param_deltas) == len(self.params)), 'unflattened array is of a wrong size'
        if not inplace:
            temp_params = self.params
        for i, param in enumerate(self.param_groups[0]['params']):

            if inplace:
                if self.defaults['problem'] != 'matrix_completion' \
                        or param.used:

                    idx = param.used_id if \
                        self.defaults['problem'] == 'matrix_completion' \
                        else i
                    param.data.add_(param_deltas[idx])

            else:
                if self.defaults['problem'] != 'matrix_completion' \
                        or param.used:

                    idx = param.used_id if \
                        self.defaults['problem'] == 'matrix_completion' \
                        else i

                    temp_params[i].data.add_(param_deltas[idx])

        if not inplace:
            return temp_params

    def m(self, g_, x):
        delta_ = x - flatten_tensor_list(self.params)
        # @
        # @
        return (torch.dot(g_.t(), delta_) + \
               0.5 * torch.dot(self.hessian_vector_product(delta_).t(), delta_) + \
               (self.defaults['sigma'] / 6) * delta_.norm(p=2) ** 3).detach()

    def m_grad(self, g_, delta_):
        return (g_ + \
               self.hessian_vector_product(delta_) + \
               (self.defaults['sigma'] / 2) * delta_.norm(p=2) * delta_).detach()

    def m_delta(self, delta):
        grad_term = torch.dot(self.grad, delta)  # @
        hess_term = 0.5 * torch.dot(self.hessian_vector_product(delta), delta)  # @
        reg_term = self.defaults['sigma'] / 6 * delta.norm(p=2) ** 3
        return (grad_term + \
                hess_term + \
                reg_term).detach()


    def beta_adapt(self, f_grad_delta, delta_):
        return ((f_grad_delta).norm(p=2) \
               / (delta_).norm(p=2)).detach()

    def cubic_subsolver(self):

        self.grad, self.params = self.get_grads_and_params()
        print('grad and params are loaded, grad dim, norm = ', self.grad.size(), self.grad.norm(p=2), len(self.params))
        beta = self.defaults.get('beta_lipschitz') if self.defaults.get('beta_lipschitz') is not None\
            else np.sqrt(self.get_hessian_eigen())
        self.least_eig = self.get_hessian_eigen(which='least', maxIter=5)
        print('least_eig', self.least_eig)

        if self.defaults['AdaHess']:
            self.t_grad += 1

            print("AdaHess grad before", self.t_grad, self.grad.norm(p=2))
            self.g_ = (self.b_1 * self.g_ + (1 - self.b_1) * self.grad).detach()

            g_hat = (self.g_ / (1 - self.b_1 ** self.t_grad)).detach()

            self.grad = g_hat
            print("AdaHess grad after", self.grad.norm(p=2), self.grad.size(),)

        self.grad_norms = self.grad.norm(p=2).detach().cpu().numpy()

        print('hessian eigenvalue is calculated', beta)

        grad_norm = self.grad.norm(p=2).detach()
        print('grad norm ', grad_norm.detach().cpu().numpy(), beta ** 2 / self.defaults['sigma'])

        eps_ = 0.5
        r = np.sqrt(self.defaults['grad_tol'] / (9 * self.defaults['sigma']))
        # ToDo: Check this constant (now beta ** 2 is changed to beta)
        self.computations_done += 1
        self.computations_done_times_samples += \
            self.defaults['sample_size_gradient']
        #if self.defaults['delta_momentum'] or grad_norm >= beta ** 2 / self.defaults['sigma']:
        self.case_n = 1
        print('case 1')
        # Get the Cauchy point
        delta = self.cauchy_point()
        #self.computations_done[-1] += self.get_num_points() + self.get_num_points('hessian')
        self.computations_done += 1
        self.computations_done_times_samples += \
            self.defaults['sample_size_hessian']

        print('delta_m ', self.m_delta(delta))
        if (not self.defaults['delta_momentum']):  # and grad_norm < beta ** 2 / self.defaults['sigma']:
            self.case_n = 2
            print('case 2')
            # Constants from the paper
            # GRADIENT DESCENT FINDS THE CUBIC-REGULARIZED NONCONVEX NEWTON STEP,
            # Carmon & Duchi, 2019

            # ToDo: scale sigma with 1/2
            if self.defaults['delta_momentum']:
                delta = torch.zeros(self.grad.size())
            sigma_ = (self.defaults['sigma'] ** 2 * r ** 3 * eps_) / (144 * (beta + 2 * self.defaults['sigma'] * r))
            eta = self.defaults.get('eta') if self.defaults.get('eta') is not None \
                else 1 / (20 * beta)

            print('generating sphere random sample, dim = ', self.grad.size()[0])
            unif_sphere = sigma_ * torch.squeeze(sample_spherical(1, ndim=self.grad.size()[0]))
            # ToDo: should I use the perturbation ball?
            ##g_ = self.m_grad(self.grad, delta) #+ 2*unif_sphere
            g_ = (self.grad + unif_sphere).detach()



            print('sphere random sample is generated')
            T_eps = self.defaults['n_iter']  # int(beta / (np.sqrt(self.defaults['sigma'] * self.defaults['grad_tol'])))
            if self.defaults['subproblem_solver'] == 'adaptive':
                # We know Lipschitz constant
                lambda_ = 1 / beta
                #lambda_ = 1
                theta = np.infty
                #f_grad_old = g_
                f_grad_old = self.m_grad(g_, delta)
                delta_old = delta
                delta = delta - lambda_ * f_grad_old

            print('Run iterations, cubic subsolver')
            # ToDo: too many iterations

            if self.defaults['innerAdam']:
                m = 0
                v = 0

            for i in range(int(T_eps)):
                print(i, '/', T_eps)
                self.computations_done += 1
                self.computations_done_times_samples += \
                    self.defaults['sample_size_hessian']
                if self.defaults['subproblem_solver'] == 'adaptive':
                    # Experimental, Nesterov’s accelerated gradient descent
                    # Accelerated Gradient Descent Escapes Saddle Points
                    # Faster than Gradient Descent
                    if self.defaults['AccGD']:
                        theta_acc = 1 / (4*np.sqrt(beta / np.sqrt(self.defaults['grad_tol']*self.defaults['sigma'])))
                        v_t = delta - delta_old
                        delta = delta + (1 - theta_acc) * v_t
                    # Update lambda
                    lambda_old = lambda_

                    """
                    self.zero_grad()
                    x = self.param_groups[0]['params']
    
                    if self.is_matrix_completion:
                        data_ = index_to_params(data, x)
                        self.loss_fn(self.model(data_), target, data_[0], data_[1]).backward(create_graph=True)
                    elif self.is_mnist or self.is_AE:  # and self.first_hv:
                        outputs = self.model(data)
                        self.loss_fn(outputs, target).backward(create_graph=True)
                        # self.first_hv = False
                    elif self.is_w_function:  # and self.first_hv:
                        self.model(x[0]).backward(create_graph=True)
                        self.perturb()
                    
                        
                    g_, _ = self.get_grads_and_params()
                    """
                    # f_grad_new = g_
                    f_grad_new = self.m_grad(g_, delta)
                    if self.defaults['innerAdam']:
                        m = self.b_1 * m + (1 - self.b_1) * f_grad_new
                        v = self.b_2 * v + (1 - self.b_2) * f_grad_new ** 2
                        m_hat = m / (1 - self.b_1 ** (i+1))
                        v_hat = v / (1 - self.b_2 ** (i+1))

                    f_grad_delta = f_grad_new - f_grad_old
                    f_grad_old = f_grad_new
                    #beta_k = self.beta_adapt(f_grad_delta, delta - delta_old).detach().numpy()
                    #lambda_ = min(np.sqrt(1 + theta) * lambda_, 1 / (lambda_ * beta**2) + 1 / (2 * beta_k**2))
                    #print('params ', beta_k, (delta - delta_old).norm(p=2))
                    #print('deltas ', delta, delta_old, f_grad_delta)
                    lambda_ = min(np.sqrt(1 + theta) * lambda_, (delta - delta_old).norm(p=2).detach().cpu().numpy() /
                                 (2 * f_grad_delta.norm(p=2).detach().cpu().numpy()))
                    delta_old = delta

                    #print('lambdas ', lambda_old, lambda_)
                    #old_delta = delta
                    if self.defaults['innerAdam']:
                        #delta = delta - lambda_ * (m_hat / (torch.sqrt(v_hat) + self.epsilon))
                        update = lambda_ * (m_hat / (torch.sqrt(v_hat) + self.epsilon))
                    else:
                        #delta = delta - lambda_ * f_grad_new
                        update = lambda_ * f_grad_new
                    print('lambdas ', lambda_, lambda_old, (delta - delta_old).norm(p=2))

                    theta = lambda_ / (lambda_old + 1e-5)
                    #print('delta_m = ', self.m_delta(delta))
                    # Empirical rule
                    #if self.is_AE or self.is_mnist:
                    #    print(abs(self.m_delta(delta)), abs(self.test_losses))
                    #    if abs(self.m_delta(delta)) > abs(self.test_losses) and i > 2:
                    #        print('delta_m has been increasing too much')
                    #        return delta, self.m_delta(delta)
                    #else:
                    #    print(abs(self.m_delta(delta)))

                elif self.defaults['subproblem_solver'] == 'non-adaptive':
                    hvp = self.hessian_vector_product(delta)
                    print('Non-adaptive', g_.norm(p=2), hvp.norm(p=2), delta.norm(p=2))
                    update = eta * (
                            g_ +
                            hvp +
                            (self.defaults['sigma'] / 2) * delta.norm(p=2) * delta
                    ).detach()
                elif self.defaults['subproblem_solver'] == 'minres':
                    n = delta.size()[0]
                    hvp = LinearOperator((n, n), matvec=self.numpy_hessian_vector_product)
                    delta_new, exitCode = minres(hvp, -g_, shift=
                    -((self.defaults['sigma'] / 2) * delta.norm(p=2)).detach().numpy())
                    update = delta - torch.from_numpy(delta_new)
                    #self.grad = self.grad.to(torch.float64)
                    print(exitCode, update.norm(p=2), update)
                    #exit(0)
                update_norm = update.norm(p=2)
                # ToDo - improve / remove empirical checks
                if False and (not self.defaults['AdaHess']) and (update_norm >= 2.5 * self.mean_update) and self.mean_update != 0:
                    print('update has been increasing too much!', self.mean_update, update_norm)
                    self.running_update = 0
                    break
                self.running_update += update_norm
                delta_old = delta
                delta = delta - update

                if (delta - delta_old).norm(p=2) < 1e-5:
                    print('no improvement anymore', (delta - delta_old).norm(p=2))
                    self.running_update = 0
                    break

                self.y_onehot.zero_()
                if 'LIN_REG' in self.defaults['problem']:
                    self.y_onehot.scatter_(1, self.defaults['target'].view(-1, 1), 1)

                previous_f = self.loss_fn(
                    self.model(
                        self.defaults['train_data']
                    ),
                    self.y_onehot if 'LIN_REG' in self.defaults['problem'] else self.defaults['target']).detach()
                self.update_params(delta)
                current_f = self.loss_fn(
                    self.model(
                        self.defaults['train_data']
                    ),
                    self.y_onehot if 'LIN_REG' in self.defaults['problem'] else self.defaults['target']).detach()


                print('delta_m = ', self.m_delta(delta), delta.norm(p=2), (delta - delta_old).norm(p=2),
                      update.norm(p=2), self.mean_update, previous_f-current_f)
                self.update_params(-delta)

            # Adaptive rho as in "Sub-sampled Cubic Regularization for Non-convex Optimization"
            if self.defaults['adaptive_rho']:
                self.computations_done += 3
                # cost of function evaluation
                self.computations_done_times_samples += \
                    2 * self.defaults['sample_size_gradient']
                # cost of m_delta evaluation
                self.computations_done_times_samples += \
                    self.defaults['sample_size_hessian']
                previous_f = self.loss_fn(
                    self.model(
                        self.defaults['train_data']
                    ),
                    self.defaults['target']).detach()
                self.update_params(delta)
                current_f = self.loss_fn(
                    self.model(
                        self.defaults['train_data']
                    ),
                    self.defaults['target']).detach()
                self.update_params(-delta)

                m_delta = self.m_delta(delta).detach()
                rho = (previous_f - current_f) / (previous_f - m_delta)
                print('adaptive rho: data', previous_f, current_f, m_delta,
                      self.defaults['sigma'])

                if rho >= self.defaults['eta_1']:
                    print('adaptive_rho: bigger that eta_1 - update')
                else:
                    print('adaptive_rho: smaller that eta_1 - no update')
                    delta = torch.zeros(self.grad.size())

                if rho > self.defaults['eta_2']:
                    print('adaptive_rho: bigger that eta_2 - decrease sigma')
                    self.defaults['sigma'] = max(min(self.defaults['sigma'], grad_norm), 1e-16)
                elif self.defaults['eta_2'] >= rho >= self.defaults['eta_1']:
                    print('adaptive_rho: bigger that eta_1, '
                          'less than eta_2 - do not update sigma')
                    pass
                else:
                    print('adaptive_rho: less that eta_1 and no improvement - increase sigma',
                          rho, self.defaults['eta_1'])
                    self.defaults['sigma'] = min(self.defaults['gamma'] \
                                             * self.defaults['sigma'], 1000)

            if self.running_update != 0:
                self.mean_update = self.running_update / int(T_eps)
                self.running_update = 0
                    #self.computations_done[-1] += self.get_num_points('hessian')
                #self.computations_done[-1] += self.get_num_points()
            if (update_norm >= 1e3 * self.mean_update):
                print('cauchy point has been increasing too much!', self.mean_update, update_norm)
                delta = torch.zeros(self.grad.size())

        return delta, self.m_delta(delta)

    def cubic_final_subsolver(self):

        self.grad, self.params = self.get_grads_and_params()
        beta = self.get_hessian_eigen()

        # In the original ARC paper they use the Cauchy Point to calculate starting delta
        delta = torch.zeros(self.grad.size())
        g_m = self.grad
        eta = 1 / (20 * beta)

        if self.defaults['subproblem_solver'] == 'adaptive':
            # We know Lipschitz constant
            lambda_ = 1 / beta
            theta = np.infty
            x = flatten_tensor_list(self.params) - lambda_ * self.grad

        while g_m.norm(p=2).detach().cpu().numpy() > self.defaults['grad_tol'] / 2:
            if self.defaults['subproblem_solver'] == 'adaptive':
                # Update Lipschitz constant
                f_grad_new = self.m_grad(g_m, x)
                g_m = f_grad_new
                beta = self.beta_adapt(f_grad_new, x)

                # Update lambda
                lambda_old = lambda_
                lambda_ = min(np.sqrt(1 + theta) * lambda_, 1 / (lambda_ * beta.detach().cpu().numpy() ** 2))
                delta -= lambda_ * f_grad_new
                x -= lambda_ * f_grad_new
                print('lambdas ', lambda_, lambda_old)
                theta = lambda_ / (lambda_old + 1e-5)
            else:
                delta -= eta * g_m
                g_m = self.grad + \
                      self.hessian_vector_product(delta) + \
                      (self.defaults['sigma'] / 2) * delta.norm(p=2)

        return delta

    def model_update(self, delta, delta_m):

        """
        x = self.param_groups[0]['params']

        if self.is_matrix_completion:
            u, v = index_to_params(self.defaults['train_data'], x)
            previous_f = self.loss_fn(
                self.model(
                    x,
                    self.defaults['train_data']
                ),
                self.defaults['target'],
                u, v)

            previous_f_ = self.loss_fn(
                self.model(
                    x,
                    self.defaults['train_data'],
                ),
                self.defaults['target'],
                u, v)

        elif self.is_mnist or self.is_AE:
            previous_f = self.loss_fn(
                self.model(
                    self.defaults['train_data']
                ),
                self.defaults['target'])

            previous_f_ = self.loss_fn(
                self.model(
                    self.defaults['train_data']
                ),
                self.defaults['target'])

        elif self.is_w_function:
            previous_f = self.model(x[0])
            previous_f_ = self.model(x[0])

        params_old = flatten_tensor_list(self.get_grads_and_params()[1])
        """
        self.update_params(delta)
        """
        x = self.param_groups[0]['params']

        if self.is_matrix_completion:
            current_f = self.loss_fn(
                self.model(
                    x,
                    self.defaults['train_data']
                ),
                self.defaults['target'],
                u, v)
        elif self.is_mnist or self.is_AE:
            current_f = self.loss_fn(
                self.model(
                    self.defaults['train_data']
                ),
                self.defaults['target'])
        elif self.is_w_function:
            current_f = self.model(x[0])

        params_new = flatten_tensor_list(self.get_grads_and_params()[1])
        print('change params', params_old.norm(p=2), params_new.norm(p=2), (params_old - params_new).norm(p=2))
        print('prev f', previous_f, previous_f_)
        print('curr f', current_f)

        # Momentum as in the paper
        # Z. Want et al. Cubic Regularization with Momentum for Nonconvex Optimization. AUAI, 2018.
        if False and not self.is_matrix_completion and \
                self.step_old is not None and current_f < previous_f:
            grad_new, params_new = self.get_grads_and_params()
            params_new = flatten_tensor_list(params_new)
            momentum_stepsize = min(momentum_const, grad_new.norm(p=2), delta.norm(p=2))
            print('momentum stepsize', momentum_stepsize)
            y_new = flatten_tensor_list(params_new)
            v_new = y_new + momentum_stepsize * (y_new - self.step_old)
            print('params before ', flatten_tensor_list(self.get_grads_and_params()[1]))
            self.update_params(-params_new)
            print('params 0 ', flatten_tensor_list(self.get_grads_and_params()[1]))
            self.update_params(v_new)
            print('params mid ', flatten_tensor_list(self.get_grads_and_params()[1]))
            if self.is_w_function:
                v_f = self.loss_fn(self.param_groups[0]['params'][0])
            else:
                v_f = self.loss_fn(self.model(self.defaults['train_data']), self.defaults['target'])
            print('v f', v_f)
            if v_f < current_f:
                current_f = v_f
                delta = v_new - params_old
            else:
                self.update_params(params_new - v_new)
            print('params after ', flatten_tensor_list(self.get_grads_and_params()[1]))

        function_decrease = previous_f - current_f
        model_decrease = -delta_m

        print(function_decrease, model_decrease)
        # ToDo: it is originally without abs
        rho = function_decrease / abs(model_decrease)
        print('rho =', rho)

        #assert (model_decrease >= 0), 'negative model decrease. This should not have happened'
        if self.case_n == 1:
            print('Case 1', delta_m, -function_decrease,
                  max(delta_m, -function_decrease),
                  -np.sqrt(self.defaults['grad_tol']**3 / self.defaults['sigma']))
            if max(delta_m, -function_decrease).detach().numpy() > \
                   -np.sqrt(self.defaults['grad_tol']**3 / self.defaults['sigma']):
                'Case 1 is not satisfied!'
                self.defaults['double_sample_size'] = True


        # Update x if step delta is successful
        if rho >= self.defaults['eta_1']:
            # We assume only one group for now
            print('Successful iteration', self.defaults['sigma'])
            grad_old, params_old = self.get_grads_and_params()
            self.step_old = flatten_tensor_list(params_old)
            #self.update_params(delta)
        else:
            # ToDo: fix problematic updates with small batch size and Cauchy - super risky!
            if self.case_n == 2 or function_decrease < 0:
                # Temp
                if False and function_decrease < 0:
                    self.update_params(-delta)

        # Update the penalty parameter rho (in the code it's sigma) if adaptive_rho = True.
        # It is so by default
        if self.defaults['adaptive_rho']:
            # Update sigma (penalty parameter) if step delta is very successful
            if rho >= self.defaults['eta_2']:
                self.defaults['sigma'] = max(self.defaults['sigma'] / self.defaults['gamma'], 1e-16)
                # alternative (Cartis et al. 2011): sigma = max(min(grad_norm,sigma), np.nextafter(0,1))
                print('Very successful iteration', self.defaults['sigma'])

            elif rho < self.defaults['eta_1']:
                if self.case_n == 2:
                    self.defaults['sigma'] = self.defaults['sigma'] * self.defaults['gamma']
                print('Unsuccessful iteration', self.defaults['sigma'])
        """
    def numpy_hessian_vector_product(self, v):
        input = torch.from_numpy(v).to(torch.float32)
        output = self.hessian_vector_product(input)\
            .detach().cpu().numpy()
        return output

    def hessian_vector_product(self, v):
        """
        compute the hessian vector product of Hv, where
        gradsH is the gradient at the current point,
        params is the corresponding variables,
        v is the vector.
        """
        end3 = time.time()


        # Compute Hessian on the same sample of points
        if not self.is_w_function:
            try:
                data, target = next(self.defaults['dataloader_iterator_hess'])
            except Exception as e:  #StopIteration:
                print('exception', str(e))
                dataloader_iterator_hess = iter(self.defaults['train_loader_hess'])
                data, target = next(dataloader_iterator_hess)

            if self.is_AE:
                data = Variable(data.view(data.size(0), -1))
                target = data

        if not self.is_w_function:
            data = data.to(self.defaults['dev'])
            target = target.to(self.defaults['dev'])

        # if self.first_hv:
        self.zero_grad()
        x = self.param_groups[0]['params']

        if self.is_matrix_completion:
            data_ = index_to_params(data, x)
            self.loss_fn(self.model(data_), target, data_[0], data_[1]).backward(create_graph=True)
        elif self.is_classification or self.is_AE:  # and self.first_hv:
            outputs = self.model(data)
            self.y_onehot_hess.zero_()
            if 'LIN_REG' in self.defaults['problem']:
                self.y_onehot_hess.scatter_(1, target.view(-1, 1), 1)
            #print(outputs, self.y_onehot)
            self.loss_fn(outputs, self.y_onehot_hess if 'LIN_REG' in self.defaults['problem'] else target)\
                .backward(create_graph=True)
            #self.loss_fn(outputs, target).backward(create_graph=True)
            # self.first_hv = False
        elif self.is_w_function:  # and self.first_hv:
            self.model(x[0]).backward(create_graph=True)
            self.perturb()
            # self.first_hv = False


        gradsh, params = self.get_grads_and_params()
        #print('grads ', len(gradsh), gradsh)
        #print('params ', len(params), params)
        #print(len(gradsh), len(params), v.shape)
        try:
            v_temp = v.clone()
        except:
            v_temp = np.copy(v)
        # ToDo - super dangerous, only for matrix_completion
        if self.is_matrix_completion:
            if len(gradsh) < len(v):
                v_temp = v[:len(gradsh)]

            elif len(gradsh) > len(v):
                v_temp = torch.cat((v, torch.zeros(len(gradsh) - len(v))))

            if self.defaults['AdaHess']:
                self.t_hess += 1
                b = torch.distributions.bernoulli.Bernoulli(torch.tensor([0.5]))
                rad = (2 * b.sample(self.grad.size()) - 1).squeeze()
                D = rad * torch.autograd.grad(gradsh, data_, grad_outputs=rad,
                                     only_inputs=True, retain_graph=True)
                print('D', self.t_hess, D.size(), D)
                self.D = (self.b_2 * self.D + (1 - self.b_2) * D ** 2).detach()
                hv = torch.sqrt((self.D / (1 - self.b_2 ** self.t)))
            else:
                hv = torch.autograd.grad(gradsh, data_, grad_outputs=v_temp,
                                     only_inputs=True, retain_graph=True)

        else:
            if self.defaults['AdaHess']:
                print('Start AdaHess for hvp')
                self.t_hess += 1
                # Temporal & spatial averaging
                b = torch.distributions.bernoulli.Bernoulli(torch.tensor([0.5]))
                rad = (2 * b.sample(self.grad.size()) - 1).squeeze()
                D = torch.autograd.grad(gradsh, params, grad_outputs=rad,
                                     only_inputs=True, retain_graph=True)
                D = (rad * flatten_tensor_list(D))
                n_ = len(D)

                # Before the code-block was here
                print('D', D.norm(p=2))
                self.D = (self.b_2 * self.D + (1 - self.b_2) * D ** 2).detach()
                print('D', D.norm(p=2))
                hv = torch.sqrt((self.D / (1 - self.b_2 ** self.t_hess)))

                # Now the code-block is here (D -> hv)
                b = self.defaults['block_size']
                if b is not None:
                    hv = torch.cat((hv[:b * (n_ // b)].reshape(-1, b)
                                   .mean(1).repeat_interleave(b), hv[b * (n_ // b):])) \
                        * v_temp
                else:
                    hv = hv * v_temp

                print('hv', hv.norm(p=2))
                print('Start AdaHess for hvp')
            else:
                print('hvp')
                hv = torch.autograd.grad(gradsh, params, grad_outputs=v_temp,
                                     only_inputs=True, retain_graph=True)
        if self.is_w_function:
            hv = self.perturb(type_='hessian', hv=hv[0])

        #print(torch.cat(hv).unique(dim=0).shape)
        if self.is_matrix_completion:
            rank = self.defaults['rank']
            hv_shape = torch.cat(hv).unique(dim=0).shape
            hv_unique_shape = hv_shape[0] * hv_shape[1]
            #print(hv_unique_shape, v.shape[0])
            if hv_unique_shape < v.shape[0]:
                till = int((v.shape[0] - hv_unique_shape) / rank)
                hv = torch.cat(
                    (torch.cat(hv).unique(dim=0),
                        torch.zeros(till, rank)
                     )
                )

            elif hv_unique_shape > v.shape[0]:
                hv = torch.cat(hv).unique(dim=0)[:int(v.shape[0] / rank), :]

            else:
                hv = torch.cat(hv).unique(dim=0)
        #print(hv.shape)
        #print(hv.unique(dim=0).shape)
        #print(hv[0].shape, hv)
        #print(hv.shape)
        print('hess vec product time: ', time.time() - end3)
        return hv if self.defaults['AdaHess'] else flatten_tensor_list(hv)


