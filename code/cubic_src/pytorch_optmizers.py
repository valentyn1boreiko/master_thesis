import utils
import numpy as np
import torch


class SRC(utils.SRCutils):
    def __init__(self, *args, **kwargs):
        super(SRC, self).__init__(*args, **kwargs)

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

        print('Cubic subsolver')
        delta, delta_m = self.cubic_subsolver()

        # Momentun, experimental
        if self.defaults['delta_momentum']:
            self.t += 1
            print('delta', delta)
            self.m = self.b_1 * self.m + (1 - self.b_1) * delta
            self.v = self.b_2 * self.v + (1 - self.b_2) * delta**2
            m_hat = self.m / (1 - self.b_1**self.t)
            v_hat = self.v / (1 - self.b_2**self.t)
            delta = self.defaults['delta_momentum_stepsize'] \
                * (m_hat / (torch.sqrt(v_hat) + self.epsilon))

        self.model_update(delta, delta_m)
        self.samples_seen[-1] += self.get_num_points() + self.get_num_points('hessian')


        # Check if we are doing enough progress
        print('final accuracy ', -1/100 * np.sqrt(self.defaults['grad_tol']**3 / self.defaults['sigma']))
        # ToDo: check if condition delta_m <= 0 is required
        if 0 >= delta_m >= -1/100 * np.sqrt(self.defaults['grad_tol']**3 / self.defaults['sigma']):
            print('do cubic final subsolver')
            delta = self.cubic_final_subsolver()
            self.param_groups[0]['params'].data.add_(delta)

        return loss
