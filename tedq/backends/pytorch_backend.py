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


tcomplex = torch.complex64
INV_SQRT2 = torch.tensor(1.0 / math.sqrt(2), dtype=torch.float64)

PI = torch.tensor(math.pi, dtype=torch.float64)


class PyTorchBackend(CompiledCircuit):
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

        super().__init__(backend, circuit, use_cotengra = use_cotengra, use_jdopttn = use_jdopttn, tn_mode=tn_mode, hyper_opt = hyper_opt, tn_simplify = tn_simplify)

        self._tn_simplify = tn_simplify

        self._requires_grad = kwargs.get("requires_grad", True)
        self._interface = kwargs.get("interface", "pytorch")
        self._diff_method = kwargs.get("diff_method", "back_prop")
        


        if self._interface != "pytorch":
            raise ValueError(
                f'{self._interface}: pytroch_backend only supports pytorch interface!'
            )


    def check_parameters_torch_device(self, params):
        """An auxiliary function to check the Torch device specified for
        the input patameters.
        
        Args:
            params (list[torch.tensor]): list of parameters to check
        """
        
        the_same = None
        for par in params:

            #print(par.device)
            # Using hasattr in case par is not type of torch tensor
            try:
                if par.is_cuda:
                    index = par.device.index
                else:
                    # put -1 if it is not in GPU
                    index = -1

                if the_same is None:
                    the_same = index

                if the_same != index:
                    raise ValueError("input parameters are not in the same device!!")

            except AttributeError as error:
                raise ValueError("input parameters must be type of pytorch tensor!!") from error
                print(error)

        # some quantum circuits do not need any input parameters
        if params:
            self._device = params[0].device
        #print("1:  ", self._device)



    def __call__(self, *params):
        '''
        internal call function
        '''
        # print(params)
        # check wether all parameters are in the same device
        self.check_parameters_torch_device(params)
        # update the device for this class
        self.update_device(self._device)

        # No need to keep track of gradient information
        if self._requires_grad is False:
            # print("NO grad")
            with torch.no_grad():
                return self.execute(*params)

        # diff_method {"back_prop", "param_shift", "adjoint", "reversible"}
        else:

            if self._diff_method == "back_prop":
                # print("back_prop grad")
                return self.execute(*params)

            if self._diff_method == "param_shift":
                dtype_list = [param.dtype for param in params]
                len_dtype = len(dtype_list)
                for i in range(len_dtype):
                    if i == (len_dtype-1):
                        break
                    if dtype_list[i] == dtype_list[i+1]:  # pylint: disable=no-else-continue
                        continue
                    else:
                        raise ValueError("input parameters for parameter shift method must have the same data type!")

                return self.param_shift_execute(*params)

            if self._diff_method == "finite_diff":
                return self.finite_diff_execute(*params)

            raise Exception(
                f"Differentiation method {self._diff_method} is not supported. "
                f"Supported methods include {{back_prop, param_shift, finite_diff }}"
            )

    def param_shift_execute(self, *params):
        '''
        Execute quantum circuit and get the gradient of the parameters with parameter shift method
        '''
        run_kwargs = {}
        run_kwargs["jacobian"] = self.jacobian_param_shift
        run_kwargs["backend"] = self
        new_params = tuple()
        for par in params:
            for element in par.view(-1):
                new_params = new_params + (element,)
        return TorchExecute.apply(run_kwargs, *new_params)

    def jacobian_param_shift(self, params):
        '''
        Get jacobian vectors of quantum circuits by the parameter shift rule
        '''
        data_type = params[0].dtype
        count = 0
        jac = None
        for idx, trnblepars in self._trainableparams.items():
            if len(trnblepars) > 0:
                for j in range(len(trnblepars)):
                    param_shift = self._operators[idx].get_parameter_shift()
                    grad = 0.
                    for _c, _a, _s in param_shift:
                        #print(_c, _a, _s)
                        #print(params)
                        #print(_c, _a, _s)
                        # since _a is eaqual to 1., it is equivalent to 
                        # new_param = [p for p in list(params)]
                        # new_param[count + j] *= _c
                        # params is the list of trainable params in order
                        new_param = [_a * p for p in list(params)]
                        new_param[count + j] += _s
                        #print(new_param)
                        grad += _c * self.execute(*new_param)

                        if jac is None:
                            jac = torch.zeros(
                                (len(grad), len(params)), dtype=data_type, device = self._device
                            )
                    jac[:, count + j] = grad
                count += len(trnblepars)

        return torch.as_tensor(jac, dtype=data_type)

    def finite_diff_execute(self, *params):
        '''
        Execute quantum circuit and get the gradient of the parameters with finite differential method.
        '''
        raise NotImplementedError

    def get_trainable_parameters(self):
        '''
        return trainable parameters
        '''
        raise NotImplementedError



    def execute(self, *params):
        '''
        Execute quantum circuit and get the gradient of the parameters with back-propagation method.
        '''

        new_params = tuple()
        for par in params:
            for element in par.view(-1, 1):
                new_params = new_params + (element,)
                # print(element.requires_grad)
        #print("input parameters:   ", new_params)
        super().execute(*new_params)


        if self._use_cotengra:
            self._parse_circuit_cotengra()
            results = []
            for i, _ in enumerate(self.measurements):
                arrays = []
                zero_state = [torch.tensor([1.,0.], dtype=tcomplex, device = self._device) for _ in range(self._num_qubits)]
                arrays.extend(zero_state)
                arrays.extend(self._operands)
                if self.measurements[i].return_type is Expectation:
                    arrays.append(self._tensor_of_gate(self.measurements[i].obs.name, []))
                    arrays.extend(self._adjointoperands)
                    arrays.extend(zero_state)
                    result = self._optimize_order_trees[i].contract(arrays, prefer_einsum = True, backend='torch')
                    result = torch.squeeze(result.real)
                    results.append(result)
                if self.measurements[i].return_type is Probability:
                    arrays.extend(self._adjointoperands)
                    arrays.extend(zero_state)
                    result = self._optimize_order_trees[i].contract(arrays, prefer_einsum = True, backend='torch')
                    result = torch.squeeze(result.real)
                    results.append(result)
                if self.measurements[i].return_type is State:
                    arrays.extend(zero_state)
                    result = self._optimize_order_trees[i].contract(arrays, prefer_einsum = True, backend='torch')
                    results.append(result)


        elif self._use_jdopttn:
            self._parse_circuit_cotengra()
            #print("2: ", self._device)
            #if 1:
            #    for tensor in self._operands:
            #        print(tensor.device)
            #    print("......")
            #    for tensor in self._operands:
            #        print(tensor.device)
            results = []
            for i, _ in enumerate(self.measurements):
                arrays = []
                zero_state = [torch.tensor([1.,0.], dtype=tcomplex, device=self._device) for _ in range(self._num_qubits)]
                arrays.extend(zero_state)
                arrays.extend(self._operands)
                if self.measurements[i].return_type is Expectation:
                    arrays.append(self._tensor_of_gate(self.measurements[i].obs.name, []))
                    arrays.extend(self._adjointoperands)
                    arrays.extend(zero_state)
                    if self._tn_simplify:
                        arrays = self._tensor_networks[i].simplify_arrays(arrays)
                    result = self._optimize_order_trees[i].contract(arrays, backend='torch')
                    result = torch.squeeze(result.real)
                    results.append(result)
                if self.measurements[i].return_type is Probability:
                    arrays.extend(self._adjointoperands)
                    arrays.extend(zero_state)
                    if self._tn_simplify:
                        arrays = self._tensor_networks[i].simplify_arrays(arrays)
                    result = self._optimize_order_trees[i].contract(arrays, backend='torch')
                    result = torch.squeeze(result.real)
                    results.append(result)
                if self.measurements[i].return_type is State:
                    arrays.extend(zero_state)
                    if self._tn_simplify:
                        arrays = self._tensor_networks[i].simplify_arrays(arrays)
                    result = self._optimize_order_trees[i].contract(arrays, backend='torch')
                    results.append(result)


        else: 
            self._parser_circuit()
            initstate = self.get_initstate()
            self._operands.append(initstate)
            axeslist = deepcopy(self._axeslist)
            permutationlist = deepcopy(self._permutationlist)   

            for _ in range(len(self._operands) - 1):
                # print("deal with ?th gate: ", i)
                statevector = self._operands.pop(-1)
                appliedgate = self._operands.pop(-1)
                axes = axeslist.pop(-1)
                perms = permutationlist.pop(-1) 
                # print(statevector.shape)  
                # print(axes)
                # print(appliedgate)    
                # print(appliedgate.requires_grad)
                newstate = torch.tensordot(appliedgate, statevector, axes)
                # print(newstate.requires_grad) 
                # print(newstate.shape)
                newstate = newstate.permute(perms)
                self._operands.append(newstate) 
            results = self.get_measurement_results(self._operands[0])


        # print(results)
        #output must be a torch.tensor form
        try:
            results = torch.stack(results, 0)
        except:
            raise ValueError(f'You can not have multiple measurements with different shapes!!')

        return results

    def get_measurement_results(self, state):
        '''
        get measurement rusults based on final quantum circuit state
        '''
        results = []
        for meas in self.measurements:
            if meas.return_type is Expectation:
                perms = (
                    list(range(1, meas.obs.qubits[0] + 1))
                    + [0]
                    + list(range(meas.obs.qubits[0] + 1, self._num_qubits))
                )
                tmpt = self._tensor_of_gate(meas.obs.name, [])
                tmpt = torch.tensordot(
                    tmpt, state, dims=([1], meas.obs.qubits)
                )  # order need to change!
                tmpt = tmpt.permute(perms)
                axes = list(range(self._num_qubits))
                tmpt = torch.tensordot(torch.conj(state), tmpt, dims=(axes, axes))
                result = torch.squeeze(tmpt.real)
                results.append(result)

            if meas.return_type is Probability:
                if meas.qubits is None:
                    results.append(torch.abs(state) ** 2)
                else:
                    probs_tensor = torch.abs(state) ** 2
                    axes = list(range(self._num_qubits))
                    axes = [x for x in axes if x not in meas.qubits]
                    if len(axes) != 0:
                        result = torch.sum(probs_tensor, dim=axes)
                    else:
                        result = probs_tensor
                    results.append(result)

            if meas.return_type is State:
                results.append(state)

        return results

    def get_initstate(self):
        if self._init_state:
            matrix = deepcopy(self._init_state.matrix)
            _b = torch.from_numpy(matrix)
            _b = _b.type(tcomplex)
            _b = _b.to(self._device)
            shape = [2 for _ in range(self._num_qubits)]
            shape = tuple(shape)
            return _b.reshape(shape)
        else:
            return self.default_initstate()

    # in state |00...0>
    def default_initstate(self):
        r'''
        
        Initial all :math:`|0\rangle` state

        '''
        shape = [2 for _ in range(self._num_qubits)]
        _b = torch.zeros(shape, dtype=tcomplex, device = self._device)
        _b.view(1, -1).data[0][0] = 1.0 + 0j
        return _b

    def complex_conjugate(self, ts):
        '''
        Get complex conjugate of that tensor
        '''
        
        shape = ts.shape
        shape_tensor = torch.tensor(shape)
        prod_shape = torch.prod(shape_tensor)
        new_size = int(torch.sqrt(prod_shape))
        new_shape = (new_size, new_size)

        ts = ts.reshape(new_shape)
        ts = ts.conj()
        ts = ts.T
        ts = ts.reshape(shape)
        
        return ts#.conj().T

    @property
    def device(self):
        r'''
        return the device used for calculation, GPU or CPU
        '''
        return self._device

    @classmethod
    def update_device(cls, device):
        cls._device = device
    

    def _matrix_to_tensor(self, matrix, num_qubits):
        '''
        converting a numpy matrix to corresponding backend tensor
        '''

        ts = torch.from_numpy(matrix)
        ts = ts.type(tcomplex)
        ts = ts.to(self._device)
        shape = [2 for _ in range(2*num_qubits)]
        shape = tuple(shape)
        return ts.reshape(shape)

    @classmethod
    def get_I_tensor(cls, paramslist):
        '''
        Get corresponding tensor. .

        See :meth:`~tedq.Backends.compiled_circuit.CompiledCircuit.get_I_tensor` in ``CompiledCircuit`` class for more detailed information.
        '''
        if len(paramslist) != 0:
            print("This gate does not need any parameters.")

        return tensor(
            [
                [1, 0], 
                [0, 1]
            ], 
            dtype=tcomplex,
            device = cls._device
        )

    @classmethod
    def get_Hadamard_tensor(cls, paramslist):
        '''
        Get corresponding tensor.

        See :meth:`~tedq.Backends.compiled_circuit.CompiledCircuit.get_Hadamard_tensor` in ``CompiledCircuit`` class for more detailed information.
        '''
        if len(paramslist) != 0:
            print("This gate does not need any parameters.")

        return tensor(
            [
                [INV_SQRT2, INV_SQRT2], 
                [INV_SQRT2, -INV_SQRT2]
            ], 
            dtype=tcomplex,
            device = cls._device
        )

    @classmethod
    def get_PauliX_tensor(cls, paramslist):
        '''
        Get corresponding tensor.

        See :meth:`~tedq.Backends.compiled_circuit.CompiledCircuit.get_PauliX_tensor` in ``CompiledCircuit`` class for more detailed information.
        '''
        if len(paramslist) != 0:
            print("This gate does not need any parameters.")

        return tensor(
            [
                [0, 1], 
                [1, 0]
            ], 
            dtype=tcomplex,
            device = cls._device
        )

    @classmethod
    def get_PauliY_tensor(cls, paramslist):
        '''
        Get corresponding tensor.

        See :meth:`~tedq.Backends.compiled_circuit.CompiledCircuit.get_PauliY_tensor` in ``CompiledCircuit`` class for more detailed information.
        '''
        if len(paramslist) != 0:
            print("This gate does not need any parameters.")

        return tensor(
            [
                [0j, -1j], 
                [1j, 0j]
            ], 
            dtype=tcomplex,
            device = cls._device
        )

    @classmethod
    def get_PauliZ_tensor(cls, paramslist):
        '''
        Get corresponding tensor.

        See :meth:`~tedq.Backends.compiled_circuit.CompiledCircuit.get_PauliZ_tensor` in ``CompiledCircuit`` class for more detailed information.
        '''
        if len(paramslist) != 0:
            print("This gate does not need any parameters.")

        return tensor(
            [
                [1, 0], 
                [0, -1]
            ], 
            dtype=tcomplex,
            device = cls._device
        )

    @classmethod
    def get_S_tensor(cls, paramslist):
        '''
        Get corresponding tensor.

        See :meth:`~tedq.Backends.compiled_circuit.CompiledCircuit.get_S_tensor` in ``CompiledCircuit`` class for more detailed information.
        '''
        if len(paramslist) != 0:
            print("This gate does not need any parameters.")

        return tensor(
            [
                [1, 0], 
                [0, 1j]
            ], 
            dtype=tcomplex,
            device = cls._device
        )

    @classmethod
    def get_T_tensor(cls, paramslist):
        '''
        Get corresponding tensor.

        See :meth:`~tedq.Backends.compiled_circuit.CompiledCircuit.get_T_tensor` in ``CompiledCircuit`` class for more detailed information.
        '''
        if len(paramslist) != 0:
            print("This gate does not need any parameters.")

        return tensor(
            [
                [1, 0], 
                [0, torch.exp(1j * PI / 4)]
            ], 
            dtype=tcomplex,
            device = cls._device
        )

    @classmethod
    def get_SX_tensor(cls, paramslist):
        '''
        Get corresponding tensor.

        See :meth:`~tedq.Backends.compiled_circuit.CompiledCircuit.get_SX_tensor` in ``CompiledCircuit`` class for more detailed information.
        '''
        if len(paramslist) != 0:
            print("This gate does not need any parameters.")

        return tensor(
            [
                [0.5 + 0.5j, 0.5 - 0.5j], 
                [0.5 - 0.5j, 0.5 + 0.5j]
            ], 
            dtype=tcomplex,
            device = cls._device
        )

    @classmethod
    def get_CNOT_tensor(cls, paramslist):
        '''
        Get corresponding tensor.

        See :meth:`~tedq.Backends.compiled_circuit.CompiledCircuit.get_CNOT_tensor` in ``CompiledCircuit`` class for more detailed information.
        '''
        if len(paramslist) != 0:
            print("This gate does not need any parameters.")

        return tensor(
            [
                [1, 0, 0, 0], 
                [0, 1, 0, 0], 
                [0, 0, 0, 1], 
                [0, 0, 1, 0]
            ], 
            dtype=tcomplex,
            device = cls._device
        ).reshape(2, 2, 2, 2)

    @classmethod
    def get_CZ_tensor(cls, paramslist):
        '''
        Get corresponding tensor.

        See :meth:`~tedq.Backends.compiled_circuit.CompiledCircuit.get_CZ_tensor` in ``CompiledCircuit`` class for more detailed information.
        '''
        if len(paramslist) != 0:
            print("This gate does not need any parameters.")

        return tensor(
            [
                [1, 0, 0, 0], 
                [0, 1, 0, 0], 
                [0, 0, 1, 0], 
                [0, 0, 0, -1]
            ], 
            dtype=tcomplex,
            device = cls._device
        ).reshape(2, 2, 2, 2)

    @classmethod
    def get_CY_tensor(cls, paramslist):
        '''
        Get corresponding tensor.

        See :meth:`~tedq.Backends.compiled_circuit.CompiledCircuit.get_CY_tensor` in ``CompiledCircuit`` class for more detailed information.
        '''
        if len(paramslist) != 0:
            print("This gate does not need any parameters.")

        return tensor(
            [
                [1, 0, 0, 0], 
                [0, 1, 0, 0],  
                [0, 0, 0, -1j], 
                [0, 0, 1j, 0]
            ], 
            dtype=tcomplex,
            device = cls._device
        ).reshape(2, 2, 2, 2)

    @classmethod
    def get_SWAP_tensor(cls, paramslist):
        '''
        Get corresponding tensor.

        See :meth:`~tedq.Backends.compiled_circuit.CompiledCircuit.get_SWAP_tensor` in ``CompiledCircuit`` class for more detailed information.
        '''
        if len(paramslist) != 0:
            print("This gate does not need any parameters.")

        return tensor(
            [
                [1, 0, 0, 0], 
                [0, 0, 1, 0], 
                [0, 1, 0, 0], 
                [0, 0, 0, 1]
            ], 
            dtype=tcomplex,
            device = cls._device
        ).reshape(2, 2, 2, 2)

    @classmethod
    def get_CSWAP_tensor(cls, paramslist):
        '''
        Get corresponding tensor.

        See :meth:`~tedq.Backends.compiled_circuit.CompiledCircuit.get_CSWAP_tensor` in ``CompiledCircuit`` class for more detailed information.
        '''
        if len(paramslist) != 0:
            print("This gate does not need any parameters.")

        return tensor(
            [
                [1, 0, 0, 0, 0, 0, 0, 0],
                [0, 1, 0, 0, 0, 0, 0, 0],
                [0, 0, 1, 0, 0, 0, 0, 0],
                [0, 0, 0, 1, 0, 0, 0, 0],
                [0, 0, 0, 0, 1, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 1, 0],
                [0, 0, 0, 0, 0, 1, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 1],
            ],
            dtype=tcomplex,
            device = cls._device
        ).reshape(2, 2, 2, 2, 2, 2)

    @classmethod
    def get_Toffoli_tensor(cls, paramslist):
        '''
        Get corresponding tensor.

        See :meth:`~tedq.Backends.compiled_circuit.CompiledCircuit.get_Toffoli_tensor` in ``CompiledCircuit`` class for more detailed information.
        '''
        if len(paramslist) != 0:
            print("This gate does not need any parameters.")

        return tensor(
            [
                [1, 0, 0, 0, 0, 0, 0, 0],
                [0, 1, 0, 0, 0, 0, 0, 0],
                [0, 0, 1, 0, 0, 0, 0, 0],
                [0, 0, 0, 1, 0, 0, 0, 0],
                [0, 0, 0, 0, 1, 0, 0, 0],
                [0, 0, 0, 0, 0, 1, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 1],
                [0, 0, 0, 0, 0, 0, 1, 0],
            ],
            dtype=tcomplex,
            device = cls._device
        ).reshape(2, 2, 2, 2, 2, 2)

    @classmethod
    def get_RX_tensor(cls, paramslist):
        '''
        Get corresponding tensor.

        See :meth:`~tedq.Backends.compiled_circuit.CompiledCircuit.get_RX_tensor` in ``CompiledCircuit`` class for more detailed information.
        '''
        theta = paramslist[0]
        _c = torch.cos(theta / 2.0)
        _js = 1j * torch.sin(-theta / 2.0)
        #return_tensor = torch.zeros([4], dtype=tcomplex,
        #    device = cls._device)
        #print("torch RX before data:  ", return_tensor.device)
        data = [
            _c, 
            _js, 
            _js, 
            _c
        ]  # [[c, js], [js, c]]
        data = torch.cat(data, dim=0)
        return_tensor = torch.as_tensor(data, dtype=tcomplex,
            device = cls._device)
        #for idx, element in enumerate(data):
        #    return_tensor[idx] = element
        #print("torch RX device:  ", return_tensor.device)
        #print("RX self._device:  ", self._device)
        return return_tensor.reshape([2, 2])

    @classmethod
    def get_RY_tensor(cls, paramslist):
        '''
        Get corresponding tensor.

        See :meth:`~tedq.Backends.compiled_circuit.CompiledCircuit.get_RY_tensor` in ``CompiledCircuit`` class for more detailed information.
        '''
        theta = paramslist[0]
        _c = torch.cos(theta / 2.0)
        _s = torch.sin(theta / 2.0)
        #return_tensor = torch.zeros([4], dtype=tcomplex,
        #    device = cls._device)
        data = [
            _c, 
            -_s, 
            _s, 
            _c
        ]  # [[c, -s], [s, c]]
        data = torch.cat(data, dim=0)
        return_tensor = torch.as_tensor(data, dtype=tcomplex,
            device = cls._device)
        #for idx, element in enumerate(data):
        #    return_tensor[idx] = element
        return return_tensor.reshape([2, 2])

    @classmethod
    def get_RZ_tensor(cls, paramslist):
        '''
        Get corresponding tensor.

        See :meth:`~tedq.Backends.compiled_circuit.CompiledCircuit.get_RZ_tensor` in ``CompiledCircuit`` class for more detailed information.
        '''
        theta = paramslist[0]
        _p = torch.exp(-0.5j * theta)
        #return_tensor = torch.zeros([4], dtype=tcomplex,
        #    device = self._device)
        data = [
            _p, 
            0.0, 
            0.0, 
            _p.conj()
        ]  # [[p, 0], [0, p.conjugate()]]
        return_tensor = torch.as_tensor(data, dtype=tcomplex,
            device = cls._device)
        #return_tensor[0] = data[0]
        #return_tensor[3] = data[3]
        #data = torch.cat(data, dim=0)
        #return_tensor = torch.as_tensor(data, dtype=tcomplex)
        #for idx, element in enumerate(data):
        #    return_tensor[idx] = element
        return return_tensor.reshape([2, 2])

    @classmethod
    def get_Rot_tensor(cls, paramslist):
        '''
        Get corresponding tensor of Rot gate

        See :meth:`~tedq.Backends.compiled_circuit.CompiledCircuit.get_Rot_tensor` in ``CompiledCircuit`` class for more detailed information.
        '''
        theta = paramslist[0]
        phi = paramslist[1]
        omega =paramslist[2]
        _c = torch.cos(phi / 2.)
        _s = torch.sin(phi / 2.)
        #return_tensor = torch.zeros([4], dtype=tcomplex,
        #    device = cls._device)
        data = [
            torch.exp(-0.5j * (theta + omega)) * _c, 
            -torch.exp(0.5j * (theta - omega)) * _s, 
            torch.exp(-0.5j * (theta - omega)) * _s, 
            torch.exp(0.5j * (theta + omega)) * _c
        ]
        data = torch.cat(data, dim=0)
        return_tensor = torch.as_tensor(data, dtype=tcomplex,
            device = cls._device)
        #for idx, element in enumerate(data):
        #    return_tensor[idx] = element
        return return_tensor.reshape([2,2])

    @classmethod
    def get_PhaseShift_tensor(cls, paramslist):
        '''
        Get corresponding tensor.

        See :meth:`~tedq.Backends.compiled_circuit.CompiledCircuit.get_PhaseShift_tensor` in ``CompiledCircuit`` class for more detailed information.
        '''
        phi = paramslist[0]
        #return_tensor = torch.zeros([4], dtype=tcomplex,
        #    device = self._device)
        data = [
            1.0,
            0.0,
            0.0,
            torch.exp(1.0j * phi),
        ]  # [[1, 0], [0, torch.exp(1j * phi)]]
        return_tensor = torch.as_tensor(data, dtype=tcomplex,
            device = cls._device)
        #return_tensor[3] = data[3]
        #data = torch.cat(data, dim=0)
        #return_tensor = torch.as_tensor(data, dtype=tcomplex)
        #for idx, element in enumerate(data):
        #    return_tensor[idx] = element
        return return_tensor.reshape([2, 2])

    @classmethod
    def get_ControlledPhaseShift_tensor(cls, paramslist):
        '''
        Get corresponding tensor.

        See :meth:`~tedq.Backends.compiled_circuit.CompiledCircuit.get_ControlledPhaseShift_tensor` in `Compliedfor more detailed information.Circuit` class 
        '''
        phi = paramslist[0]
        #return_tensor = torch.zeros([16], dtype=tcomplex,
        #    device = self._device)
        data = [
            1.0,
            0.0,
            0.0,
            0.0,
            0.0,
            1.0,
            0.0,
            0.0,
            0.0,
            0.0,
            1.0,
            0.0,
            0.0,
            0.0,
            0.0,
            torch.exp(1.0j * phi),
        ]  # [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, torch.exp(1j * phi)]]
        return_tensor = torch.as_tensor(data, dtype=tcomplex,
            device = cls._device)
        #return_tensor[15] = data[15]
        #data = torch.cat(data, dim=0)
        #return_tensor = torch.as_tensor(data, dtype=tcomplex)
        #for idx, element in enumerate(data):
        #    return_tensor[idx] = element
        return return_tensor.reshape([2, 2, 2, 2])

    @classmethod
    def get_CRX_tensor(cls, paramslist):
        '''
        Get corresponding tensor.

        See :meth:`~tedq.Backends.compiled_circuit.CompiledCircuit.get_CRX_tensor` in ``CompiledCircuit`` class for more detailed information.
        '''
        theta = paramslist[0]
        _c = torch.cos(theta / 2.0)
        _js = 1.0j * torch.sin(-theta / 2.0)
        #return_tensor = torch.zeros([16], dtype=tcomplex,
        #    device = self._device)
        data = [
            1.0,
            0.0,
            0.0,
            0.0,
            0.0,
            1.0,
            0.0,
            0.0,
            0.0,
            0.0,
            _c,
            _js,
            0.0,
            0.0,
            _js,
            _c,
        ]  # [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, c, js], [0, 0, js, c]]
        return_tensor = torch.as_tensor(data, dtype=tcomplex,
            device = cls._device)
        #return_tensor[10] = data[10]
        #return_tensor[11] = data[11]
        #return_tensor[14] = data[14]
        #return_tensor[15] = data[15]
        #data = torch.cat(data, dim=0)
        #return_tensor = torch.as_tensor(data, dtype=tcomplex)
        #for idx, element in enumerate(data):
        #    return_tensor[idx] = element
        return return_tensor.reshape([2, 2, 2, 2])

    @classmethod
    def get_CRY_tensor(cls, paramslist):
        '''
        Get corresponding tensor.

        See :meth:`~tedq.Backends.compiled_circuit.CompiledCircuit.get_CRY_tensor` in ``CompiledCircuit`` class for more detailed information.
        '''
        theta = paramslist[0]
        _c = torch.cos(theta / 2.0)
        _s = torch.sin(theta / 2.0)
        #return_tensor = torch.zeros([16], dtype=tcomplex,
        #    device = self._device)
        data = [
            1.0,
            0.0,
            0.0,
            0.0,
            0.0,
            1.0,
            0.0,
            0.0,
            0.0,
            0.0,
            _c,
            -_s,
            0.0,
            0.0,
            _s,
            _c,
        ]  # [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, c, -s], [0, 0, s, c]]
        return_tensor = torch.as_tensor(data, dtype=tcomplex,
            device = cls._device)
        #return_tensor[10] = data[10]
        #return_tensor[11] = data[11]
        #return_tensor[14] = data[14]
        #return_tensor[15] = data[15]
        #data = torch.cat(data, dim=0)
        #return_tensor = torch.as_tensor(data, dtype=tcomplex)
        #for idx, element in enumerate(data):
        #    return_tensor[idx] = element
        return return_tensor.reshape([2, 2, 2, 2])

    @classmethod
    def get_CRZ_tensor(cls, paramslist):
        '''
        Get corresponding tensor.

        See :meth:`~tedq.Backends.compiled_circuit.CompiledCircuit.get_CRZ_tensor` in ``CompiledCircuit`` class for more detailed information.
        '''
        theta = paramslist[0]
        #return_tensor = torch.zeros([16], dtype=tcomplex, device = self._device)
        data = [
            1,
            0.0,
            0.0,
            0.0,
            0.0,
            1.0,
            0.0,
            0.0,
            0.0,
            0.0,
            torch.exp(-0.5j * theta),
            0.0,
            0.0,
            0.0,
            0.0,
            torch.exp(0.5j * theta),
        ]
        return_tensor = torch.as_tensor(data, dtype=tcomplex,
            device = cls._device)
        #return_tensor[10] = data[10]
        #return_tensor[15] = data[15]

        #data = torch.cat(data, dim=0)
        #return_tensor = torch.as_tensor(data, dtype=tcomplex)
        #for idx, element in enumerate(data):
        #    return_tensor[idx] = element
        return return_tensor.reshape([2, 2, 2, 2])


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
