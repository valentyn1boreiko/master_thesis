from torch.optim.optimizer import Optimizer
import copy


class SRC(Optimizer):
    def __init__(self, params, adaptive_rho=True, subproblem_solver='adaptive',
                 batchsize_mode='fixed', opt=None):

        defaults = dict(grad_tol=opt.get('grad_tol', 1e-6),
                        adaptive_rho=adaptive_rho,
                        subproblem_solver=subproblem_solver,
                        batchsize_mode=batchsize_mode,
                        sample_size_Hessian=opt.get('sample_size_Hessian', 10),
                        sample_size_gradient=opt.get('sample_size_gradient', 100),
                        eta_1=opt.get('success_treshold', 0.1),
                        eta_2=opt.get('very_success_treshold', 0.9),
                        gamma=opt.get('penalty_increase_decrease_multiplier', 2.),
                        sigma=opt.get('initial_penalty_parameter', 1.),
                        n_iterations=opt.get('n_iterations', 100)
                        )
        super(SRC, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(SRC, self).__setstate__(state)

    def step(self, closure=None):
        """ Performs a single optimization step.
        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:

            for p in group['params']:
                if p.grad is None:
                    continue
                # average gradients or use the weighted sampling

                # calculate Hessian explicitly and
                # calculate the largest eigenvalue of the Hessian -> beta

                # differentiate two cases depending on the norm
                # of the grad

                # first case - Cauchy point

                # second case - gradient descent or adaptive gradient descent
                d_p = p.grad.data
                param_state = self.state[p]

        return
