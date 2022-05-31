#   Copyright 2021-2024 Jingdong Digits Technology Holding Co.,Ltd.
#
#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.


r"""
This module contains the :class:`GradientDescentOptimizer` for
 tuning parameters according to their gradients.
"""

# pylint: disable=line-too-long, trailing-whitespace, too-many-lines, too-many-instance-attributes, too-few-public-methods

from copy import deepcopy
import jax


class GradientDescentOptimizer:
    r'''
    Basic gradient-descent optimizer.
    Base class for other gradient-descent-based optimizers.
    
    A step of the gradient descent optimizer computes the new values via the rule
    
    .. math::
        x^{(t+1)} = x^{(t)} - \eta \nabla f(x^{(t)}).
    
    where :math:`\eta` is a user-defined hyperparameter corresponding to step size.
    
    Args:
        stepsize (float): the user-defined hyperparameter :math:`\eta`
    '''

    def __init__(self, objective_fn, trainable_params, stepsize=0.01, interface=None):
        if interface is None:
            raise ValueError(
                f'{interface}: please specify the interface, it should be the same as objective_fn'
            )
        self._interface = interface
        self._trainable_params = trainable_params
        self._stepsize = stepsize
        self.apply_grad_jit = self.apply_grad  #jax.jit(self.apply_grad)
        self.grad_func_jit = jax.grad(objective_fn, trainable_params)  #jax.jit(jax.grad(objective_fn, trainable_params))
        self.objective_fn = objective_fn

    def update_stepsize(self, stepsize):
        r"""Update the initialized stepsize value :math:`\eta`.
        This allows for techniques such as learning rate scheduling.
        
        Args:
            stepsize (float): the user-defined hyperparameter :math:`\eta`
        """
        self._stepsize = stepsize

    def step(self, *args, **kwargs):
        """Update trainable arguments with one step of the optimizer.
        
        Args:
            objective_fn (function): the objective function for optimization
            
            args : Variable length argument list for objective function
            
            grad_fn (function): optional gradient function of the objective function with respect to the variables ``x``. If ``None``, the gradient function is computed automatically. Must return the same shape of tuple [array] as the autograd derivative.
            
            kwargs : variable length of keyword arguments for the objective function
        
        Returns:
            list [array]: the new variable values :math:`x^{(t+1)}`. If single arg is provided, list [array] is replaced by array.
        """

        # g = self.compute_grad(*args, **kwargs)
        if self._interface == "jax":
            grad = self.compute_grad_jaxoptimal(*args, **kwargs)
            new_args = self.apply_grad_jit(grad, *args)
            return new_args
        if self._interface == "pytorch":
            params = list(args)
            res = self.objective_fn(*args, **kwargs)
            res.backward()
            grad = []
            for i in self._trainable_params:
                grad.append(params[i].grad)
            out_args = self.apply_grad(grad, *args)
            new_args = []
            for element in out_args:
                # This is important, so that previous grad will not affect this time's calculation
                new_ele = element.detach()
                new_ele.requires_grad = True
                new_args.append(new_ele)
            #print(new_args[0].requires_grad_())
            return tuple(new_args)
        
        raise ValueError(f'Unknown interface {self._interface}!')



    def compute_grad(self, *args):
        r"""Compute gradient of the objective function at the given point and return it along with the objective function forward pass (if available).
        
        Args:
            objective_fn (function): the objective function for optimization
            
            args (tuple): tuple of NumPy arrays containing the current parameters for the objection function
            
            kwargs (dict): keyword arguments for the objective function
            
            grad_fn (function): optional gradient function of the objective function with respect to the variables ``args``. If ``None``, the gradient function is computed automatically. Must return the same shape of tuple [array] as the autograd derivative.
        
        Returns:
            tuple (array): NumPy array containing the gradient :math:`\nabla f(x^{(t)})` and the objective function output. If ``grad_fn`` is provided, the objective function will not be evaluted and instead ``None`` will be returned.
        """
        gradient = []
        old_args = list(args)
        # print(args)
        # print(old_args)

        for i, _ in enumerate(old_args):
            new_args = deepcopy(old_args)
            # print(type(new_args[i]))
            # print(new_args[i])
            new_args[i] = old_args[i] + 0.001
            new_args = tuple(new_args)
            gradient.append(
                (self.objective_fn(*new_args) - self.objective_fn(*args)) / 0.001
            )
        return gradient

    def compute_grad_jaxoptimal(self, *args):
        r"""Compute gradient of the objective function at the given point and return it along with
        the objective function forward pass (if available).
        
        Args:
            objective_fn (function): the objective function for optimization
            
            args (tuple): tuple of NumPy arrays containing the current parameters for the objection function

            kwargs (dict): keyword arguments for the objective function

            grad_fn (function): optional gradient function of the objective function with respect to the variables ``args``. If ``None``, the gradient function is computed automatically. Must return the same shape of tuple [array] as the autograd derivative.
        
        Returns:
            tuple (array): NumPy array containing the gradient :math:`\nabla f(x^{(t)})` and the objective function output. If ``grad_fn`` is provided, the objective function will not be evaluted and instead ``None`` will be returned.

        """
        gradient = list(self.grad_func_jit(*args))
        return gradient

    def apply_grad(self, grad, *args):
        r"""Update the variables to take a single optimization step. Flattens and unflattens
        the inputs to maintain nested iterables as the parameters of the optimization.
        
        Args:
            grad (tuple [array]): the gradient of the objective
                function at point :math:`x^{(t)}`: :math:`\nabla f(x^{(t)})`

            args (tuple): the current value of the variables :math:`x^{(t)}`
        
        Returns:
            list [array]: the new values :math:`x^{(t+1)}`

        """
        new_args = []
        args = list(args)
        for i, _ in enumerate(args):
            if i in self._trainable_params:
                new_args.append(args[i] - self._stepsize * grad[i])
            else:
                new_args.append(args[i])

        return tuple(new_args)
