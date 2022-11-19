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

r'''
This is the backend embeded the PyTorch computation module into tedq.
 The backend is used for the computation of simulation of quantum circuit,
  as well as finding the gradient of output result among the input parameters.

There're three supported methods to calculate the gradient
    * Parameters-shift method
    * Finite-differential
    * Back-propagation method
'''

# pylint: disable=line-too-long, trailing-whitespace, too-many-lines, too-many-instance-attributes, too-few-public-methods, too-many-arguments
# pylint: disable=no-member, no-name-in-module, too-many-public-methods, too-many-statements, too-many-branches, too-many-locals

import math
from copy import deepcopy
import torch
from torch import tensor
from tedq.QInterpreter.operators.measurement import Expectation, Probability, State
from .compiled_circuit import CompiledCircuit
from .pytorch_backend import PyTorchBackend


tcomplex = torch.complex64
INV_SQRT2 = torch.tensor(1.0 / math.sqrt(2), dtype=torch.float64)

PI = torch.tensor(math.pi, dtype=torch.float64)



class QUDIOBackend(PyTorchBackend):
    r'''
    pytorch backend to do the calculation.
    '''

    # None will use cpu
    # class variable
    _device = None
    '''
    This is the backend embeded the PyTorch computation module into tedq.

    Args:
        backend (string): Name of the computation backend -- ``jax`` or ``pytorch``
        circuit (.Circuit): Circuit to be computed.
        use_cotengra (Bool): Whether to use cotengra optimizer or not.
        use_jdopttn (Bool): Whether to use cotengra optimizer or not.
        hyper_opt (dict): TODO slice options
        kwargs (dict): Other keyword arguments
    '''
    #param-shift method only works for float inputs and outputs, not complex64

    #pytorch backend only support list of measurements of the same dimension,
    #since finally a torch.stack function is used to combine all the measurement result into a torch.tensor object
    def __init__(self, backend, circuit, use_cotengra = False, use_jdopttn = False, tn_mode=False, hyper_opt = None, tn_simplify=True, **kwargs):

        print("in init")
        self._tn_simplify = tn_simplify

        self._requires_grad = kwargs.get("requires_grad", True)
        self._interface = kwargs.get("interface", "pytorch")
        self._diff_method = kwargs.get("diff_method", "back_prop")
        
        super().__init__(backend, circuit, use_cotengra = use_cotengra, use_jdopttn = use_jdopttn, tn_mode=tn_mode, hyper_opt = hyper_opt, tn_simplify = tn_simplify, **kwargs)

        self._torch_model = torch.nn.DataParallel(TorchModel(super().__call__))

        self._dataset = None
        self._dataloader = None
        self._device = "cuda:0" if torch.cuda.is_available() else "cpu"

    def __call__(self, *params):
        '''
        internal call function
        '''
        # print(self._dataset)
        outputList = torch.tensor([])
        for data in self._dataloader:
            input = data.to(self._device)
            output = self._torch_model(input, *params)
            # print("Outside: input size", input.size(),
            #       "output_size", output.size(), 
            #       "output:", output)
            outputList = torch.cat((outputList, output), 0)

        return outputList


    def set_dataset(self, dataset):
        device_count = 1 #cpu
        if self._device == "cuda:0":
            device_count = torch.cuda.device_count()
        print("Device count:", device_count)
        self._dataset = TorchDataset(dataset)
        self._dataloader = torch.utils.data.DataLoader(dataset=self._dataset, batch_size=device_count, shuffle=False)

    @property
    def device(self):
        r'''
        return the device used for calculation, GPU or CPU
        '''
        return self._device
    def dataset(self):
        return self._dataset

    @classmethod
    def update_device(cls, device):
        cls._device = device


    

class TorchDataset(torch.utils.data.Dataset):

    def __init__(self, d):
        self.len = len(d)
        self.data = d

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return self.len


class TorchModel(torch.nn.Module):
    def __init__(self, fcn):
        super().__init__()
        self._fcn = fcn
    def forward(self, x_in, *params):
        x = self._fcn(x_in, *params)
        # print("In Model: input size", x_in.size(),
        #   "output size", x.size())
        return x

class TorchExecute(torch.autograd.Function):
    '''
    Custom autograd functions for parameter-shift method to provide compatibility of backpropagtion in PyTorch ML function. Use 'Function.apply' to run the function, where any arguments can be set as long as having the same format as the 'forward' function.
    '''

    @staticmethod
    def forward(ctx, input_kwargs, *input_params):  # pylint: disable=arguments-differ
        '''
        forawrd function
        '''

        ctx.data_type = input_params[0].dtype
        backend = input_kwargs["backend"]
        ctx.backend = backend
        ctx.device = backend.device
        ctx.all_params = input_params
        ctx.jacobian = input_kwargs["jacobian"]

        return backend.execute(*input_params)

    @staticmethod
    def backward(ctx, dy_):  # pylint: disable=arguments-differ
        '''
        backward function
        '''

        #print(dy_)
        dyy = torch.as_tensor(dy_, dtype=ctx.data_type, device = ctx.device)
        jac = ctx.jacobian(ctx.all_params)
        vjp = dyy @ jac
        vjp = torch.unbind(vjp.view(-1))

        return (None,) + tuple(vjp)
