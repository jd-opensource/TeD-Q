r'''
quantum natural gradient optimizer
'''
import torch
from torch.optim.optimizer import Optimizer, required

class QNG(Optimizer):
    def __init__(self, params, lr=required, momentum=0, dampening=0,
                 weight_decay1=0, weight_decay2=0, nesterov=False):
        #print(params)
        defaults = dict(lr=lr, momentum=momentum, dampening=dampening,
                        weight_decay1=weight_decay1, weight_decay2=weight_decay2, nesterov=nesterov)
        if nesterov and (momentum <= 0 or dampening != 0):
            raise ValueError("Nesterov momentum requires a momentum and zero dampening")
        super(SGD, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(SGD, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('nesterov', False)

    def step(self, metric_tensor, closure=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            weight_decay1 = group['weight_decay1']
            weight_decay2 = group['weight_decay2']
            momentum = group['momentum']
            dampening = group['dampening']
            nesterov = group['nesterov']
            

            d_p_list = []
            p_list = []
            i_list = []
            for i, p in enumerate(group['params']):
                #print(p)
                if p.grad is None:
                    continue
                d_p = p.grad.data
                if weight_decay1 != 0:
                    d_p.add_(weight_decay1, torch.sign(p.data))
                if weight_decay2 != 0:
                    d_p.add_(weight_decay2, p.data)
                if momentum != 0:
                    param_state = self.state[p]
                    if 'momentum_buffer' not in param_state:
                        buf = param_state['momentum_buffer'] = torch.zeros_like(p.data)
                        buf.mul_(momentum).add_(d_p)
                    else:
                        buf = param_state['momentum_buffer']
                        buf.mul_(momentum).add_(1 - dampening, d_p)
                    if nesterov:
                        d_p = d_p.add(momentum, buf)
                    else:
                        d_p = buf

                #p.data.add_(-group['lr'], d_p)
                i_list.append(i)
                p_list.append(p)
                d_p_list.append(d_p)
            
            metric_tensor_list = tuple([metric_tensor[i] for i in i_list])
            metric_tensor = torch.stack(metric_tensor_list)
            #print(metric_tensor)
            metric_tensor_list = tuple([metric_tensor[(slice(None), i)] for i in i_list])
            #print(metric_tensor_list)
            metric_tensor = torch.stack(metric_tensor_list, dim=1)
            #print(p_list, d_p_list)
            grad_flat = torch.tensor(d_p_list)
            #grad_flat = grad_flat.type(torch.complex64)
            metric_tensor = metric_tensor.type(torch.float)
            x_flat = torch.tensor(p_list)
            stepsize = group['lr']
            
            x_new_flat = x_flat - stepsize * torch.linalg.solve(metric_tensor, grad_flat)
            #print(x_flat.dtype, x_new_flat.dtype)
            
            j = 0
            for p in group['params']:
                if p.grad is None:
                    continue
                    
                flat_p = p.view(-1)
                for k in range(len(flat_p)):
                    flat_p.data[k] = x_new_flat[j].data
                    j = j+1
                

        return loss