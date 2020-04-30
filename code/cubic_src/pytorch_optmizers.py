from torch.optim.optimizer import Optimizer
import utils
import torch
import numpy as np


class SRC(Optimizer):
    def __init__(self, params, model, loss_fn, adaptive_rho=True, subproblem_solver='adaptive',
                 batchsize_mode='fixed', opt=None):

        if opt is None:
            opt = dict()

        self.model = model
        self.loss_fn = loss_fn
        self.grad, self.params = None, None

        self.defaults = dict(grad_tol=opt.get('grad_tol', 1e-6),
                             adaptive_rho=adaptive_rho,
                             subproblem_solver=subproblem_solver,
                             batchsize_mode=batchsize_mode,
                             sample_size_hessian=opt.get('sample_size_Hessian', 0.005),
                             sample_size_gradient=opt.get('sample_size_gradient', 0.05),
                             eta_1=opt.get('success_treshold', 0.1),
                             eta_2=opt.get('very_success_treshold', 0.9),
                             gamma=opt.get('penalty_increase_decrease_multiplier', 2.),
                             sigma=opt.get('initial_penalty_parameter', 1.),
                             n_iterations=opt.get('n_iterations', 100),
                             target=None
                             )

        super(SRC, self).__init__(params, self.defaults)

    def __setstate__(self, state):
        super(SRC, self).__setstate__(state)

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
        return utils.flatten_tensor_list(grads), params

    def m(self, g_, x):
        delta_ = x - self.param_groups[0]['params'].data
        return g_.t() @ delta_ +\
               0.5 * utils.hessian_vector_product(self.grad, self.params, delta_) @ delta_ +\
               (self.defaults['sigma'] / 6) * delta_.norm(p=2)**3

    def m_grad(self, g_, x):
        delta_ = x - self.param_groups[0]['params'].data
        return g_ + \
               utils.hessian_vector_product(self.grad, self.params, delta_) + \
               (self.defaults['sigma'] / 2) * delta_.norm(p=2) * delta_

    def m_delta(self, delta):
        return self.grad.t() @ delta + \
               0.5 * utils.hessian_vector_product(self.grad, self.params, delta) + \
               self.defaults['sigma'] / 6 * np.linalg.norm(delta, 2)

    def beta_adapt(self, f_grad_new, x):
        return (f_grad_new - self.grad).norm(p=2) \
               / (x - self.param_groups[0]['params'].data)

    def cubic_subsolver(self):
        self.grad, self.params = self.get_grads_and_params()
        beta = utils.get_hessian_eigen(self.grad, self.params)
        grad_norm = self.grad.norm(p=2)

        if grad_norm >= beta ** 2 / self.defaults['sigma']:
            # Get the Cauchy point
            delta = utils.cauchy_point(self.grad, grad_norm, self.params, self.defaults['sigma'])
        else:
            # Constants from the paper
            # GRADIENT DESCENT FINDS THE CUBIC-REGULARIZED NONCONVEX NEWTON STEP,
            # Carmon & Duchi, 2019
            r = torch.sqrt(self.defaults['grad_tol'] / (9 * self.defaults['sigma']))
            eps_ = 0.5
            # ToDo: scale sigma with 1/2
            delta = torch.zeros(self.grad.size())
            sigma_ = (self.defaults['sigma'] ** 2 * r ** 3 * eps_) / (144 * (beta + 2 * self.defaults['sigma'] * r))
            eta = 1 / (20 * beta)

            g_ = self.grad + sigma_ * utils.sample_spherical(1, ndim=self.grad.size()[0])
            T_eps = int(beta / (np.sqrt(self.defaults['sigma'] * self.defaults['grad_tol'])))
            if self.defaults['subproblem_solver'] == 'adaptive':
                # We know Lipschitz constant
                lambda_ = 1 / beta
                theta = np.infty
                x = self.param_groups[0]['params'].data - lambda_ * self.grad

            for _ in range(T_eps):
                if self.defaults['subproblem_solver'] == 'adaptive':
                    # Update Lipschitz constant
                    f_grad_new = self.m_grad(g_, x)
                    beta = self.beta_adapt(f_grad_new, x)
                    # Update lambda
                    lambda_old = lambda_
                    lambda_ = min(np.sqrt(1 + theta) * lambda_, 1 / (lambda_ * beta**2))
                    delta -= lambda_ * f_grad_new
                    x -= lambda_ * f_grad_new
                    theta = lambda_ / lambda_old
                else:
                    delta -= eta * (
                            g_ +
                            utils.hessian_vector_product(self.grad, self.params, delta) +
                            (self.defaults['sigma'] / 2) * np.linalg.norm(delta, 2)
                    )
        return delta, self.m_delta(delta)

    def cubic_final_subsolver(self):
        self.grad, self.params = self.get_grads_and_params()
        beta = utils.get_hessian_eigen(self.grad, self.params)

        # In the original ARC paper they use the Cauchy Point to calculate starting delta
        delta = torch.zeros(self.grad.size())
        g_m = self.grad
        eta = 1 / (20 * beta)

        if self.defaults['subproblem_solver'] == 'adaptive':
            # We know Lipschitz constant
            lambda_ = 1 / beta
            theta = np.infty
            x = self.param_groups[0]['params'].data - lambda_ * self.grad

        while g_m.norm(p=2) > self.defaults['grad_tol'] / 2:
            if self.defaults['subproblem_solver'] == 'adaptive':
                # Update Lipschitz constant
                f_grad_new = self.m_grad(g_m, x)
                g_m = f_grad_new
                beta = self.beta_adapt(f_grad_new, x)

                # Update lambda
                lambda_old = lambda_
                lambda_ = min(np.sqrt(1 + theta) * lambda_, 1 / (lambda_ * beta ** 2))
                delta -= lambda_ * f_grad_new
                x -= lambda_ * f_grad_new
                theta = lambda_ / lambda_old
            else:
                delta -= eta * g_m
                g_m = self.grad + \
                      utils.hessian_vector_product(self.grad, self.params, delta) + \
                      (self.defaults['sigma'] / 2) * np.linalg.norm(delta, 2)

        return delta

    def model_update(self, delta, delta_m):
        previous_f = self.model(self.param_groups[0]['params'].data)
        current_f = self.param_groups[0]['params'].data + delta

        function_decrease = previous_f - current_f
        model_decrease = -delta_m

        rho = function_decrease / model_decrease
        assert (model_decrease >= 0), 'negative model decrease. This should not have happened'

        # Update x if step delta is successful
        if rho >= self.defaults['eta_1']:
            # We assume only one group for now
            self.param_groups[0]['params'].data.add_(delta)

        # Update the penalty parameter rho (in the code it's sigma) if adaptive_rho = True.
        # It is so by default
        if self.defaults['adaptive_rho']:
            # Update sigma (penalty parameter) if step delta is very successful
            if rho >= self.defaults['eta_2']:
                self.defaults['sigma'] = max(self.defaults['sigma'] / self.defaults['gamma'], 1e-16)
                # alternative (Cartis et al. 2011): sigma = max(min(grad_norm,sigma), np.nextafter(0,1))
                print('Very successful iteration')

            elif rho < self.defaults['eta_1']:
                self.defaults['sigma'] = self.defaults['sigma'] * self.defaults['gamma']
                print('Unsuccessful iteration')

    def step(self, closure=None):
        """ Performs a single optimization step.
        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        delta, delta_m = self.cubic_subsolver()

        self.model_update(delta, delta_m)

        # Check if we are doing enough progress
        if delta_m >= -1/100 * np.sqrt(self.defaults['grad_tol']**3 / self.defaults['sigma']):
            delta = self.cubic_final_subsolver()
            self.param_groups[0]['params'].data.add_(delta)

        return loss
