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
This module contains the :class:`pytorch_interface` class, tedq can use jax to calculate the
 quantum circuit, but interfacing with torch format.
"""

# pylint: disable=line-too-long, trailing-whitespace, too-many-lines, too-many-instance-attributes, too-few-public-methods

import torch
import numpy as np
#import jax
from jax import numpy as jnp
#from jax import jacfwd, jacrev


class PytorchInterface(torch.autograd.Function):
    r"""
    This is the class that provide the PyTorch interface for JAX computation module. 
    """
    
    # apply in jax_backend
    @staticmethod
    def forward(ctx, input_kwargs, *input_):   # pylint: disable=arguments-differ
        r"""
        Customized `forward` function wrapped the JAX computation module.
        The main purpose is to convert the variable type between PyTorch and JAX.

        Args:
            dy(kwargs): TODO 
            input_(Torch.tensor): Parameters
        """

        ctx.num_intns = len(input_)

        ctx.params = [jnp.asarray(element.detach().numpy()) for element in input_]

        ctx.execute_func = input_kwargs["execute_func"]

        jax_results = ctx.execute_func(ctx.params)

        ctx.jacobian = input_kwargs["jaccobian_func"]

        # jac = ctx.jacobian(ctx.params)

        # params = list(par.flatten().squeeze().tolist() for par in input_)

        # ctx.execute_func = input_kwargs["execute_func"]

        # ctx.args = params

        # trainable_params = list(range(len(params)))

        # ctx.grad_func_jit = jax.jit(jax.grad(execute_func, trainable_params))

        # print(params)
        # jax_results = ctx.execute_func(params)
        # ctx.jacobian = jacrev(ctx.execute_func)
        # jax_results, ctx.f_vjp = jax.vjp(ctx.execute_func, *params)
        # print(jax_results)
        # results = torch.from_numpy(np.asarray(jax_results))
        # print(type(jax_result))
        results = [
            torch.from_numpy(np.asarray(element).copy()) for element in jax_results  # pylint: disable=no-member
        ]
        ctx.len_res = len(results)
        #print(results, len(results))
        # print(type(results))
        if len(results) == 1:
            results = results[0]#.squeeze()  # pylint: disable=no-member
            #print(results)
            if not results.shape:
                results = results.unsqueeze(0)
            #print(results)
        else:
            results = torch.stack(results, 0) # pylint: disable=no-member
        #jax backend pytorch interface only support list of measurements with the same dimensions.
        #the output must be a torch.tensor object
        return results
        # return results

    @staticmethod
    def backward(ctx, *dy):
        r"""
        Customized `backward` function wrapped the JAX computation module.
        The main purpose is to convert the variable type between PyTorch and JAX and interfacing the gradient calculated by each module. With this function, the back-propagation in the JAX part function are available for PyTorch module.

        Args:
            dy(Torch.tensor): Gradient from previous layer
        """

        # results = list(ctx.grad_func_jit(*ctx.args))
        # print(dy)
        dy_tensor = torch.stack(dy, 0)  # pylint: disable=no-member
        # print(dy_tensor)
        # print(ctx.params)
        results = ctx.jacobian(ctx.params)

        # Converting jacobian format from JAX to PyTorch
        d_list = [[] for _ in range(ctx.num_intns)]
        for i in range(ctx.len_res):
            for j in range(ctx.num_intns):
                d_list[j].append(results[i][j])

        for j in range(ctx.num_intns):
            d_list[j] = jnp.stack(d_list[j])
            # d_list[j] = jnp.split(d_list[j], ctx.len_res, 0)
            d_list[j] = torch.from_numpy(np.asarray(d_list[j]).copy()).squeeze()  # pylint: disable=no-member


            if d_list[j].dim() > 0:
                d_list[j] = torch.tensordot(dy_tensor, d_list[j], dims=1).squeeze()
            else:
                d_list[j] = dy_tensor * d_list[j]

            # d_list[j] = d_list[j].view(size)
            # print(d_list[j].size())

        # print(d_list)

        gradient = tuple(d_list)

        # print("below is the result!")
        # print(results)
        # print(type(results[0]))
        # np.asarray(results[0])
        # results = tuple(dy[idx]*torch.from_numpy(np.asarray(val).copy()) for idx, val in enumerate(results))
        # print(results)
        # gradient = torch.stack(results, 0)

        # print(gradient)
        # print(gradient.size())

        # gradient = tuple(gradient.split(1, 1))

        return (None,) + gradient #None ==> kwargs in forward
