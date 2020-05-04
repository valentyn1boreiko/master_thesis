import torch
import time
import numpy as np
import scipy.linalg as la
from torch.optim.optimizer import Optimizer
import matplotlib.pyplot as plt
import config


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


class SRCutils(Optimizer):
    def __init__(self, params, adaptive_rho=True, subproblem_solver='adaptive',
                 batchsize_mode='fixed', opt=None):

        if opt is None:
            opt = dict()

        self.model = opt['model']
        self.loss_fn = opt['loss_fn']
        self.case_n = 1
        self.n = opt['n']
        self.grad, self.params = None, None
        self.gradient_samples_seen = [0]
        self.test_losses = []
        self.step_old = None


        self.defaults = dict(grad_tol=opt.get('grad_tol', 1e-2),
                             adaptive_rho=adaptive_rho,
                             subproblem_solver=subproblem_solver,
                             batchsize_mode=batchsize_mode,
                             sample_size_hessian=opt.get('sample_size_hessian', 0.0001),
                             sample_size_gradient=opt.get('sample_size_gradient', 0.001),
                             eta_1=opt.get('success_treshold', 0.1),
                             eta_2=opt.get('very_success_treshold', 0.9),
                             gamma=opt.get('penalty_increase_decrease_multiplier', 2.),
                             sigma=opt.get('initial_penalty_parameter', 16.),
                             n_epochs=opt.get('n_epochs', 14),
                             target=None,
                             log_interval=opt.get('log_interval', 1)
                             )
        super(SRCutils, self).__init__(params, self.defaults)

    def get_accuracy(self, loader):
        self.model.eval()
        correct, total, loss = (0, 0, 0)

        for batch_idx_, (data_, target_) in enumerate(loader):
            # Get Samples
            data_ = data_.to(self.defaults['dev'])
            target_ = target_.to(self.defaults['dev'])
            outputs = self.model(data_)
            loss += self.loss_fn(outputs, target_).detach() * len(target_)
            # Get prediction
            _, predicted = torch.max(outputs.data, 1)
            # Total number of labels
            total += len(target_)
            # Total correct predictions
            correct += (predicted == target_).sum().detach()
            del outputs
            del predicted
        acc = 100 * correct / total
        loss = loss / total
        print("All points {}".format(total))
        return loss, acc

    def print_acc(self, train_loader, epoch, batch_idx):
        #train_loss, train_acc = self.get_accuracy(train_loader)
        #print(
        #    "Epoch {} Train Loss: {:.4f} Accuracy :{:.4f} Test Loss: {:.4f} Accuracy: {:.4f}".format(epoch, train_loss,
        #                                                                                             train_acc,
        #                                                                                             test_loss,
        #                                                                                             test_acc))

        if batch_idx % self.defaults['log_interval'] == 0:
            test_loss, test_acc = self.get_accuracy(config.test_loader)

            print(
                "Epoch {} Test Loss: {:.4f} Accuracy: {:.4f}".format(epoch,
                                                                     test_loss,
                                                                     test_acc))
            self.test_losses.append(test_loss)
            plt.plot(self.gradient_samples_seen, self.test_losses)
            plt.savefig('fig/loss_src.png')
            print('idx ', batch_idx, self.test_losses, self.gradient_samples_seen)
            self.gradient_samples_seen.append(self.gradient_samples_seen[-1])

    def cauchy_point(self, grads_norm):
        # Compute Cauchy radius
        # ToDo: replace hessian-vec product with the upper bound (beta)
        product = self.hessian_vector_product(self.grad).t() @ self.grad / (self.defaults['sigma'] * grads_norm ** 2)
        R_c = -product + torch.sqrt(product ** 2 + 2 * grads_norm / self.defaults['sigma'])
        delta = -R_c * self.grad / grads_norm
        return delta

    def get_eigen(self, H_bmm, matrix=None, maxIter=10, tol=1e-3, method='lanczos'):
        """
        compute the top eigenvalues of model parameters and
        the corresponding eigenvectors.
        """
        # change the model to evaluation mode, otherwise the batch Normalization Layer will change.
        # If you call this function during training, remember to change the mode back to training mode.
        params = self.params
        if params:
            q = flatten_tensor_list([torch.randn(p.size(), device=p.device) for p in params])
        else:
            q = torch.randn(matrix.size()[0])

        q = q / torch.norm(q)

        eigenvalue = None

        if method == 'power':
            # Power iteration
            for _ in range(maxIter):
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
                        return eigenvalue_tmp, q
                    else:
                        eigenvalue = eigenvalue_tmp
            return eigenvalue, q

        elif method == 'lanczos':
            # Lanczos iteration
            b = 0
            if params:
                q_last = flatten_tensor_list([torch.zeros(p.size(), device=p.device) for p in params])
            else:
                q_last = torch.zeros(matrix.size()[0])
            q_s = [q_last]
            a_s = []
            b_s = []
            for _ in range(maxIter):
                Hv = H_bmm(q)
                a = torch.dot(Hv, q)
                Hv -= (b * q_last + a * q)
                q_last = q
                q_s.append(q_last)
                b = torch.norm(Hv)
                a_s.append(a)
                b_s.append(b)
                if b == 0:
                    break
                q = Hv / b
            eigs, _ = la.eigh_tridiagonal(a_s, b_s[:-1])

            return max(abs(eigs))

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
            params.append(param)
            if param.grad is None:
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
                param.data.add_(param_deltas[i])
            else:
                temp_params[i].data.add_(param_deltas[i])

        if not inplace:
            return temp_params

    def m(self, g_, x):
        delta_ = x - flatten_tensor_list(self.params)
        return g_.t() @ delta_ + \
               0.5 * self.hessian_vector_product(delta_).t() @ delta_ + \
               (self.defaults['sigma'] / 6) * delta_.norm(p=2) ** 3

    def m_grad(self, g_, delta_):
        return g_ + \
               self.hessian_vector_product(delta_) + \
               (self.defaults['sigma'] / 2) * delta_.norm(p=2) * delta_

    def m_delta(self, delta):
        return self.grad.t() @ delta + \
               0.5 * self.hessian_vector_product(delta).t() @ delta + \
               self.defaults['sigma'] / 6 * delta.norm(p=2)

    def beta_adapt(self, f_grad_delta, delta_):
        return (f_grad_delta).norm(p=2) \
               / (delta_).norm(p=2)

    def cubic_subsolver(self):
        self.grad, self.params = self.get_grads_and_params()
        print('grad and params are loaded, grad dim = ', self.grad.size())
        beta = np.sqrt(self.get_hessian_eigen())
        print('hessian eigenvalue is calculated', beta)
        print(self.grad)
        grad_norm = self.grad.norm(p=2)
        print('grad norm ', grad_norm.detach().numpy(), beta ** 2 / self.defaults['sigma'])

        eps_ = 0.5
        r = np.sqrt(self.defaults['grad_tol'] / (9 * self.defaults['sigma']))
        # ToDo: Check this constant (now beta ** 2 is changed to beta)

        if grad_norm.detach().numpy() >= beta ** 2 / self.defaults['sigma']:
            self.case_n = 1
            # Get the Cauchy point
            delta = self.cauchy_point(grad_norm)
        else:
            self.case_n = 2
            # Constants from the paper
            # GRADIENT DESCENT FINDS THE CUBIC-REGULARIZED NONCONVEX NEWTON STEP,
            # Carmon & Duchi, 2019

            # ToDo: scale sigma with 1/2
            delta = torch.zeros(self.grad.size())
            sigma_ = (self.defaults['sigma'] ** 2 * r ** 3 * eps_) / (144 * (beta + 2 * self.defaults['sigma'] * r))
            eta = 1 / (20 * beta)

            print('generating sphere random sample, dim = ', self.grad.size()[0])
            unif_sphere = sigma_ * torch.squeeze(sample_spherical(1, ndim=self.grad.size()[0]))
            g_ = self.m_grad(self.grad, delta) + unif_sphere
            print('sphere random sample is generated')
            T_eps = int(beta / (np.sqrt(self.defaults['sigma'] * self.defaults['grad_tol'])))
            if self.defaults['subproblem_solver'] == 'adaptive':
                # We know Lipschitz constant
                lambda_ = 1 / beta
                #lambda_ = 1
                theta = np.infty
                f_grad_old = self.m_grad(g_, delta)
                delta_old = delta
                delta = delta - lambda_ * f_grad_old


            print('Run iterations, cubic subsolver')
            # ToDo: too many iterations

            for i in range(int(T_eps)):
                print(i, '/', T_eps)
                if self.defaults['subproblem_solver'] == 'adaptive':
                    # Update Lipschitz constant

                    # Update lambda
                    lambda_old = lambda_
                    f_grad_new = self.m_grad(g_, delta)
                    f_grad_delta = f_grad_new - f_grad_old
                    f_grad_old = f_grad_new
                    #beta_k = self.beta_adapt(f_grad_delta, delta - delta_old).detach().numpy()
                    #lambda_ = min(np.sqrt(1 + theta) * lambda_, 1 / (lambda_ * beta**2) + 1 / (2 * beta_k**2))
                    #print('params ', beta_k, (delta - delta_old).norm(p=2))
                    #print('deltas ', delta, delta_old, f_grad_delta)
                    lambda_ = min(np.sqrt(1 + theta) * lambda_, (delta - delta_old).norm(p=2).detach().numpy() /
                                 (2 * f_grad_delta.norm(p=2).detach().numpy()))
                    delta_old = delta

                    #print('lambdas ', lambda_old, lambda_)
                    old_delta = delta
                    delta = delta - lambda_ * f_grad_new
                    print('lambdas ', lambda_, lambda_old, (delta - delta_old).norm(p=2))
                    if (delta - delta_old).norm(p=2) < 1e-3:
                        print('no improvement anymore')
                        break
                    theta = lambda_ / (lambda_old + 1e-5)
                    print('delta_m = ', self.m_delta(delta))
                    # Empirical rule
                    if abs(self.m_delta(delta)) > abs(self.test_losses[0]) and i > 2:
                        print('delta_m has been increasing too much')
                        return delta, self.m_delta(delta)

                else:
                    delta -= eta * (
                            g_ +
                            self.hessian_vector_product(delta) +
                            (self.defaults['sigma'] / 2) * delta.norm(p=2) * delta
                    )
                    print('delta_m = ', self.m_delta(delta))

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

        while g_m.norm(p=2).detach().numpy() > self.defaults['grad_tol'] / 2:
            if self.defaults['subproblem_solver'] == 'adaptive':
                # Update Lipschitz constant
                f_grad_new = self.m_grad(g_m, x)
                g_m = f_grad_new
                beta = self.beta_adapt(f_grad_new, x)

                # Update lambda
                lambda_old = lambda_
                lambda_ = min(np.sqrt(1 + theta) * lambda_, 1 / (lambda_ * beta.detach().numpy() ** 2))
                delta -= lambda_ * f_grad_new
                x -= lambda_ * f_grad_new
                print('lambdas ', lambda_, lambda_old)
                theta = lambda_ / (lambda_old + 1e-5)
            else:
                delta -= eta * g_m
                g_m = self.grad + \
                      self.hessian_vector_product(delta) + \
                      (self.defaults['sigma'] / 2) * np.linalg.norm(delta, 2)

        return delta

    def model_update(self, delta, delta_m):

        previous_f = self.loss_fn(self.model(self.defaults['train_data']), self.defaults['target'])
        previous_f_ = self.loss_fn(self.model(self.defaults['train_data']), self.defaults['target'])
        params_old = flatten_tensor_list(self.get_grads_and_params()[1])
        momentum_const = 0.9
        self.update_params(delta)

        current_f = self.loss_fn(self.model(self.defaults['train_data']), self.defaults['target'])
        print('prev f', previous_f, previous_f_)
        print('curr f', current_f)

        if self.step_old is not None and current_f < previous_f:
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

    def hessian_vector_product(self, v):
        """
        compute the hessian vector product of Hv, where
        gradsH is the gradient at the current point,
        params is the corresponding variables,
        v is the vector.
        """
        end3 = time.time()
        # Compute Hessian on the same sample of points
        try:
            data, target = next(self.defaults['dataloader_iterator_hess'])
        except StopIteration:
            dataloader_iterator_hess = iter(config.train_loader_hess)
            data, target = next(dataloader_iterator_hess)
        self.zero_grad()
        self.loss_fn(self.model(data), target).backward(create_graph=True)
        gradsh, params = self.get_grads_and_params()
        hv = torch.autograd.grad(gradsh, params, grad_outputs=v,
                                 only_inputs=True, retain_graph=True)
        print('hess vec product time: ', time.time() - end3)
        #print(hv)
        return flatten_tensor_list(hv)


