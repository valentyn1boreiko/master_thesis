import utils
import numpy as np
import torch
import sys
import psutil


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
        self.first_hv = True
        print('Memory before subsolver', psutil.virtual_memory().used >> 20)
        delta, delta_m = self.cubic_subsolver()
        print('Memory after subsolver', psutil.virtual_memory().used >> 20)
        # Momentun, experimental
        if self.defaults['delta_momentum']:
            print('Memory before t', psutil.virtual_memory().used >> 20)
            self.t += 1
            print('delta', delta.detach())
            print('Memory before m', psutil.virtual_memory().used >> 20)
            self.m = (self.b_1 * self.m + (1 - self.b_1) * delta.detach()).detach()
            print('Memory before v', psutil.virtual_memory().used >> 20)
            self.v = (self.b_2 * self.v + (1 - self.b_2) * delta.detach()**2).detach()
            print('Memory before m_hat', psutil.virtual_memory().used >> 20)
            m_hat = (self.m / (1 - self.b_1**self.t)).detach()
            print('Memory before v_hat', psutil.virtual_memory().used >> 20)
            v_hat = (self.v / (1 - self.b_2**self.t)).detach()
            print('Memory before delta', psutil.virtual_memory().used >> 20)
            delta = (self.defaults['delta_momentum_stepsize'] \
                * (m_hat / (torch.sqrt(v_hat) + self.epsilon))).detach()
            print('Memory after delta', psutil.virtual_memory().used >> 20)
        
        print('Memory before update', psutil.virtual_memory().used >> 20)
        self.model_update(delta, delta_m)
        del(self.params)
        del(self.grad)
        print('Memory after update', psutil.virtual_memory().used >> 20)
        self.samples_seen += self.get_num_points() + self.get_num_points('hessian')


        # Check if we are doing enough progress
        print('final accuracy ', -1/100 * np.sqrt(self.defaults['grad_tol']**3 / self.defaults['sigma']))
        # ToDo: check if condition delta_m <= 0 is required
        if 0 >= delta_m >= -1/100 * np.sqrt(self.defaults['grad_tol']**3 / self.defaults['sigma']):
            print('do cubic final subsolver')
            delta = self.cubic_final_subsolver()
            self.param_groups[0]['params'].data.add_(delta)

        return loss
