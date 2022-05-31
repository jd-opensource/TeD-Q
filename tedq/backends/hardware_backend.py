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
Real IBMQ quantum computer backend.
"""

# pylint: disable=line-too-long, trailing-whitespace, too-many-lines, too-many-instance-attributes, too-few-public-methods, pointless-string-statement

from collections import OrderedDict
import math
import qiskit
from qiskit import QuantumCircuit, transpile, IBMQ
from qiskit.circuit import Parameter
import torch
import numpy as np

from tedq.QInterpreter.operators.measurement import Expectation, Probability, State


# Caution! now all the parameters must be trainable!!

# pylint: disable=too-many-public-methods
class HardwareBackend():
    r'''
    Class for Real IBMQ quantum computer backend.

    Args:
        backend (string): Name of the computation backend -- ``jax`` or ``pytorch``
        circuit (.Circuit): Circuit to be computed.

    '''

    # pylint: disable=too-many-arguments, too-many-locals, too-many-branches, too-many-statements
    def __init__(self, backend, circuit):

        self._device = None
        self._num_qubits = circuit.num_qubits
        self._backend = backend
        self._operatorname = OrderedDict()
        self._appliedqubits = OrderedDict()
        self._operatorparams = OrderedDict()
        self._trainableparams = OrderedDict()
        self._operators = OrderedDict()
        self._operands = []
        self._adjointoperands = []

        for operator in circuit.operators:
            # print(operator)
            self._operators[operator.instance_id] = operator
            self._operatorname[operator.instance_id] = operator.name
            self._appliedqubits[operator.instance_id] = operator.qubits
            self._operatorparams[operator.instance_id] = operator.parameters
            self._trainableparams[operator.instance_id] = operator.trainable_params
        self._measurements = circuit.measurements

        #if self._num_qubits != 1:
        #    raise ValueError("Currently hardware only support 1 qubit PauliZ measurement!")

        #if not self._measurements[0].obs.name == "PauliZ":
        #    raise ValueError("Currently hardware only support 1 qubit PauliZ measurement!")

        self._gate_to_qiskit = {
            "I": self.get_I_qiskit,
            "Hadamard": self.get_Hadamard_qiskit,
            "PauliX": self.get_PauliX_qiskit,
            "PauliY": self.get_PauliY_qiskit,
            "PauliZ": self.get_PauliZ_qiskit,
            "S": self.get_S_qiskit,
            "T": self.get_T_qiskit,
            "SX": self.get_SX_qiskit,
            "CNOT": self.get_CNOT_qiskit,
            "CZ": self.get_CZ_qiskit,
            "CY": self.get_CY_qiskit,
            "SWAP": self.get_SWAP_qiskit,
            "CSWAP": self.get_CSWAP_qiskit,
            "Toffoli": self.get_Toffoli_qiskit,
            "RX": self.get_RX_qiskit,
            "RY": self.get_RY_qiskit,
            "RZ": self.get_RZ_qiskit,
            "Rot": self.get_Rot_qiskit,
            "PhaseShift": self.get_PhaseShift_qiskit,
            "ControlledPhaseShift": self.get_ControlledPhaseShift_qiskit,
            "CRX": self.get_CRX_qiskit,
            "CRY": self.get_CRY_qiskit,
            "CRZ": self.get_CRZ_qiskit,
        }

        self._qc = QuantumCircuit(self._num_qubits, 0)
        self._params_count = 0
        self._quantum_parameters = []

        for operator in circuit.operators:
            name = operator.name
            qubits = operator.qubits
            self._gate_to_qiskit[name](qubits)

        # TODO: need to make sure each qubit only has one measurement! probs measurement measure selected qubits or all qubits

        rotations = []
        for measurement in self._measurements:
            if measurement.return_type is Expectation:
                rotations.extend(measurement.obs.diagonalizing_gates())

            elif measurement.return_type is Probability:
                pass

            else:
                raise ValueError(f'hardware backend do not support {measurement.return_type} measurement!')

        for operator in rotations:
            name = operator.name
            qubits = operator.qubits
            self._gate_to_qiskit[name](qubits)

        self._qc.measure_all()

        provider = IBMQ.get_provider('ibm-q')
        #self._backend = provider.get_backend('ibmq_qasm_simulator')
        self._backend = provider.get_backend('ibmq_belem')

        #from qiskit import Aer, transpile
        #self._backend = Aer.get_backend("aer_simulator_statevector")


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

        # some quantum circuits do not need any input parameters
        if params:
            self._device = params[0].device
        #print("1:  ", self._device)

    def __call__(self, *params):
        r'''
        '''

        self.check_parameters_torch_device(params)

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


    def execute(self, *params):
        '''
        Execute quantum circuit and get the gradient of the parameters with back-propagation method.
        '''

        new_params = tuple()
        for par in params:
            for element in par.view(-1, 1):
                element = element.detach().numpy()
                element = element[0]
                new_params = new_params + (element,)
                # print(element.requires_grad)
        #print("input parameters:   ", new_params)
        result = self.real_execute(*new_params)
        return result

    def real_execute(self, *params):
        r'''
        '''
        params_bind = dict()
        for i, p_name in enumerate(self._quantum_parameters):
            params_bind[p_name] = params[i]

        qc = self._qc.bind_parameters(params_bind)
        job = self._backend.run(transpile(qc, self._backend))
        counts = job.result().get_counts()

        self._probabilities = np.zeros(2**self._num_qubits)
        for i in range(2**self._num_qubits):
            str_bin_i = to_str_bin(i, self._num_qubits)
            p = counts.get(str_bin_i, 0)
            #print(p)
            self._probabilities[i] = p/1024.
        #print(self._probabilities)
        self._probabilities = self._probabilities.reshape([2 for _ in range(self._num_qubits)])

        probabilities = torch.from_numpy(self._probabilities)
        results = self.get_measurement_results(probabilities)

        #output must be a torch.tensor form
        try:
            results = torch.stack(results, 0)
        except:
            raise ValueError(f'You can not have multiple measurements with different shapes!!')

        return results


    def get_measurement_results(self, probabilities):
        '''
        get measurement rusults based on final quantum circuit state
        '''
        results = []
        for meas in self._measurements:
            if meas.return_type is Expectation:
                axes = list(range(self._num_qubits))
                axes = [x for x in axes if x not in meas.obs.qubits]
                #print(axes)
                #print(probabilities)
                if len(axes) != 0:
                    result = torch.sum(probabilities, dim=axes)
                else:
                    result = probabilities
                eigenvals = meas.obs.eigvals
                eigenvals = torch.from_numpy(eigenvals)
                #print(eigenvals)
                #print(result)
                result = result@eigenvals
                result = result.unsqueeze(dim=0)
                results.append(result)

            if meas.return_type is Probability:
                if meas.qubits is None:
                    results.append(probabilities)
                else:
                    axes = list(range(self._num_qubits))
                    axes = [x for x in axes if x not in meas.qubits]
                    results.append(torch.sum(probabilities, dim=axes))

            if meas.return_type is State:
                raise ValueError(f'hardware backend do not support State measurement!')

        return results

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
                        try:
                            len_grad = len(grad)
                        except TypeError as e:
                            len_grad = 1
                            print(e)

                        if jac is None:
                            jac = torch.zeros(
                                (len_grad, len(params)), dtype=data_type, device = self._device
                            )
                    jac[:, count + j] = grad
                count += len(trnblepars)

        return torch.as_tensor(jac, dtype=data_type)

    @property
    def device(self):
        r'''
        return the device used for calculation, GPU or CPU
        '''
        return self._device

    def get_I_qiskit(self, qubits):
        '''
        Get corresponding qiskit object.
        '''

        pass

    
    def get_Hadamard_qiskit(self, qubits):
        '''
        Get corresponding qiskit object.
        '''
        qubit = qubits[0]
        self._qc.h(qubit)

    
    def get_PauliX_qiskit(self, qubits):
        '''
        Get corresponding qiskit object.
        '''
        qubit = qubits[0]
        self._qc.x(qubit)

    
    def get_PauliY_qiskit(self, qubits):
        '''
        Get corresponding qiskit object.
        '''
        qubit = qubits[0]
        self._qc.y(qubit)

    
    def get_PauliZ_qiskit(self, qubits):
        '''
        Get corresponding qiskit object.
        '''
        qubit = qubits[0]
        self._qc.z(qubit)

    
    def get_S_qiskit(self, qubits):
        '''
        Get corresponding qiskit object.
        '''
        qubit = qubits[0]
        self._qc.s(qubit)

    
    def get_T_qiskit(self, qubits):
        '''
        Get corresponding qiskit object.
        '''
        qubit = qubits[0]
        self._qc.t(qubit)

    
    def get_SX_qiskit(self, qubits):
        '''
        Get corresponding qiskit object.
        '''
        qubit = qubits[0]
        self._qc.sx(qubit)

    
    def get_CNOT_qiskit(self, qubits):
        '''
        Get corresponding qiskit object.
        '''
        qubit = qubits[0]
        self._qc.cx(qubit)

    
    def get_CZ_qiskit(self, qubits):
        '''
        Get corresponding qiskit object.
        '''
        qubit = qubits[0]
        self._qc.cz(qubit)

    
    def get_CY_qiskit(self, qubits):
        '''
        Get corresponding qiskit object.
        '''
        qubit = qubits[0]
        self._qc.cy(qubit)

    
    def get_SWAP_qiskit(self, qubits):
        '''
        Get corresponding qiskit object.
        '''

        raise NotImplementedError

    
    def get_CSWAP_qiskit(self, qubits):
        '''
        Get corresponding qiskit object.
        '''

        raise NotImplementedError

    
    def get_Toffoli_qiskit(self, qubits):
        '''
        Get corresponding qiskit object.
        '''

        raise NotImplementedError

    
    def get_RX_qiskit(self, qubits):
        '''
        Get corresponding qiskit object.
        '''

        qubit = qubits[0]
        param = Parameter("rx_"+str(self._params_count))
        self._qc.rx(param, qubit)
        self._params_count += 1
        self._quantum_parameters.append(param)

    
    def get_RY_qiskit(self, qubits):
        '''
        Get corresponding qiskit object.
        '''

        qubit = qubits[0]
        param = Parameter("ry_"+str(self._params_count))
        self._qc.ry(param, qubit)
        self._params_count += 1
        self._quantum_parameters.append(param)

    
    def get_RZ_qiskit(self, qubits):
        '''
        Get corresponding qiskit object.
        '''
        qubit = qubits[0]
        param = Parameter("rz_"+str(self._params_count))
        self._qc.rz(param, qubit)
        self._params_count += 1
        self._quantum_parameters.append(param)

    
    def get_Rot_qiskit(self, qubits):
        '''
        Get corresponding qiskit object.
        '''

        raise NotImplementedError

    
    def get_PhaseShift_qiskit(self, qubits):
        '''
        Get corresponding qiskit object.
        '''

        raise NotImplementedError

    
    def get_ControlledPhaseShift_qiskit(self, qubits):
        '''
        Get ControlledPhaseShift tensor
        '''

        raise NotImplementedError

    
    def get_CRX_qiskit(self, qubits):
        '''
        Get CRX tensor
        '''

        raise NotImplementedError

    
    def get_CRY_qiskit(self, qubits):
        '''
        Get CRY tensor
        '''

        raise NotImplementedError

    
    def get_CRZ_qiskit(self, qubits):
        '''
        Get corresponding qiskit object.
        '''

        raise NotImplementedError

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

def to_str_bin(x, n_qubits):
    r'''
    converting a decimal number into binary format.

    **Example**

    .. code-block:: python3
        x = 3
        n_qubits = 3
        >>> dec_to_bin(3, 3)
        '011'
    '''
    # pylint: disable=invalid-name

    n = bin(x)[2:]
    n = n.zfill(n_qubits)
    result = [ix for ix in n]
    #print(result)
    return ''.join(result)