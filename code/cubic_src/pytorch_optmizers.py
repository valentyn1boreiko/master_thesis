from torch.optim.optimizer import Optimizer
import utils
import torch


class SRC(Optimizer):
    def __init__(self, params, adaptive_rho=True, subproblem_solver='adaptive',
                 batchsize_mode='fixed', opt=None):

        if opt is None:
            opt = dict()

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
                             n_iterations=opt.get('n_iterations', 100)
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

    def step(self, closure=None):
        """ Performs a single optimization step.
        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        grad, params = self.get_grads_and_params()
        beta = utils.get_hessian_eigen(grad, params)
        grad_norm = grad.norm(p=2)

        if grad_norm >= beta**2 / self.sigma:
            # Get the Cauchy point
            step = utils.cauchy_point(grad, grad_norm, params, self.sigma)
        else:
            # Constants from the paper
            # GRADIENT DESCENT FINDS THE CUBIC-REGULARIZED NONCONVEX NEWTON STEP,
            # Carmon & Duchi, 2019
            r = torch.sqrt(self.grad_tol/(9*self.sigma))
            eps_ = 0.5
            # ToDo: scale sigma with 1/2
            delta, sigma_, eta = 0, (self.sigma**2 * r**3 * eps_) / (144*(beta + 2*self.sigma*r)), 1 / (20*beta)
            g_ = grad + sigma_ * utils.sample_spherical(1, ndim=grad.size()[0])

        # average gradients or use the weighted sampling

        # calculate Hessian explicitly and
        # calculate the largest eigenvalue of the Hessian -> beta

        # differentiate two cases depending on the norm
        # of the grad

        # first case - Cauchy point

        # second case - gradient descent or adaptive gradient descent
        #d_p = p.grad.data
        #param_state = self.state[p]

        return loss
