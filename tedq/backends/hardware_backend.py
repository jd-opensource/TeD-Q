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

import torch
import numpy as np

from tedq.QInterpreter.operators.measurement import Expectation, Probability, State

import qiskit
from qiskit import transpile, IBMQ
from qiskit import QuantumCircuit as qiskit_QuantumCircuit
from qiskit.circuit import Parameter

from quafu import User
from quafu import QuantumCircuit as quafu_QuantumCircuit
import numpy as np

# Caution! now all the parameters must be trainable!!

# pylint: disable=too-many-public-methods
class HardwareBackend():
    r'''
    Class for Real hardware quantum computer backend.

    Args:
        backend (string): Name of the computation backend -- ``jax`` or ``pytorch``
        circuit (.Circuit): Circuit to be computed.

    '''

    # pylint: disable=too-many-arguments, too-many-locals, too-many-branches, too-many-statements
    def __init__(self, backend, circuit, num_shots = 2000):

        self._num_shots = num_shots
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

        for operator in circuit.operators:
            parameters_length = len(operator.parameters)
            trainable_parameters_length = len(operator.trainable_params)
            if parameters_length != trainable_parameters_length:
                raise ValueError("In hardware backend, all the parameters in quantum circuit must be trainable parameters!")

        self._measurements = circuit.measurements

        #if self._num_qubits != 1:
        #    raise ValueError("Currently hardware only support 1 qubit PauliZ measurement!")

        #if not self._measurements[0].obs.name == "PauliZ":
        #    raise ValueError("Currently hardware only support 1 qubit PauliZ measurement!")


        # TODO: need to make sure each qubit only has one measurement! probs measurement measure selected qubits or all qubits



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
        result = result.to(self._device)
        #print(result)
        return result

    def real_execute(self, *params):
        r'''
        '''
        raise NotImplementedError


    def get_measurement_results(self, probabilities):
        '''
        get measurement rusults based on final quantum circuit state
        '''
        results = []
        for meas in self._measurements:
            if meas.return_type is Expectation:

                if isinstance(meas.obs, list):
                    observables_axes = []
                    observables_eigens = []
                    for ob in meas.obs:
                        observables_axes.extend(ob.qubits)
                        eigen = torch.from_numpy(ob.eigvals)
                        observables_eigens.append(eigen)

                    # must be in order from small to large
                    if not sorted(observables_axes) == observables_axes:
                        raise ValueError(f'Please put the measurement observables in order! Current order is: {observables_axes}')
                    axes = list(range(self._num_qubits))
                    axes = [x for x in axes if x not in observables_axes]

                    for _ in range(len(observables_eigens) - 1):
                        # print("deal with ?th gate: ", i)
                        eigenvals = observables_eigens.pop(-1)
                        applied_eigen = observables_eigens.pop(-1)
                        eigenvals = torch.outer(applied_eigen, eigenvals)
                        observables_eigens.append(eigenvals)

                    eigenvals = observables_eigens[0]


                else:
                    axes = list(range(self._num_qubits))
                    axes = [x for x in axes if x not in meas.obs.qubits]
                    #print(axes)
                    #print(probabilities)

                    eigenvals = meas.obs.eigvals
                    eigenvals = torch.from_numpy(eigenvals)



                if len(axes) != 0:
                    result = torch.sum(probabilities, dim=axes)
                else:
                    result = probabilities

                #print(eigenvals)
                #print(result)
                result = result.view(-1)
                eigenvals = eigenvals.view(-1)
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



class HardwareBackend_qiskit(HardwareBackend):
    r'''
    Class for Real IBMQ quantum computer backend.

    Args:
        backend (string): Name of the computation backend -- ``jax`` or ``pytorch``
        circuit (.Circuit): Circuit to be computed.

    '''


    # pylint: disable=too-many-arguments, too-many-locals, too-many-branches, too-many-statements
    def __init__(self, backend, circuit):
        super().__init__(backend, circuit)

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

        self._qc = qiskit_QuantumCircuit(self._num_qubits, 0)
        self._params_count = 0
        self._quantum_parameters = []

        for operator in circuit.operators:
            name = operator.name
            qubits = operator.qubits
            self._gate_to_qiskit[name](qubits)

        # TODO: need to make sure each qubit only has one measurement! probs measurement measure selected qubits or all qubits

        for measurement in self._measurements:
            if measurement.return_type is Expectation:
                if isinstance(measurement.obs, list):
                    for ob in measurement.obs:
                        rotations.extend(ob.diagonalizing_gates())
                else:
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

    def real_execute(self, *params):
        r'''
        '''
        params_bind = dict()
        for i, p_name in enumerate(self._quantum_parameters):
            params_bind[p_name] = params[i]

        qc = self._qc.bind_parameters(params_bind)
        job = self._backend.run(transpile(qc, self._backend), shots=self._num_shots)
        counts = job.result().get_counts()

        self._probabilities = np.zeros(2**self._num_qubits)
        for i in range(2**self._num_qubits):
            str_bin_i = to_str_bin(i, self._num_qubits)
            p = counts.get(str_bin_i, 0)
            #print(p)
            self._probabilities[i] = p/self._num_shots
        #print(self._probabilities)
        self._probabilities = self._probabilities.reshape([2 for _ in range(self._num_qubits)])

        probabilities = torch.from_numpy(self._probabilities)
        results = self.get_measurement_results(probabilities)

        if len(results) == 1:
            return results[0]

        #output must be a torch.tensor form
        try:
            results = torch.stack(results, 0)
        except:
            raise ValueError(f'You can not have multiple measurements with different shapes!!')

        return results

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





class HardwareBackend_quafu(HardwareBackend):
    r'''
    Class for quafu IBMQ quantum computer backend.

    Args:
        backend (string): Name of the computation backend -- ``jax`` or ``pytorch``
        circuit (.Circuit): Circuit to be computed.

    '''


    # pylint: disable=too-many-arguments, too-many-locals, too-many-branches, too-many-statements
    def __init__(self, backend, circuit):
        super().__init__(backend, circuit)

        self._gate_to_quafu = {
            "I": self.get_I_quafu,
            "Hadamard": self.get_Hadamard_quafu,
            "PauliX": self.get_PauliX_quafu,
            "PauliY": self.get_PauliY_quafu,
            "PauliZ": self.get_PauliZ_quafu,
            "S": self.get_S_quafu,
            "T": self.get_T_quafu,
            "SX": self.get_SX_quafu,
            "CNOT": self.get_CNOT_quafu,
            "CZ": self.get_CZ_quafu,
            "CY": self.get_CY_quafu,
            "SWAP": self.get_SWAP_quafu,
            "CSWAP": self.get_CSWAP_quafu,
            "Toffoli": self.get_Toffoli_quafu,
            "RX": self.get_RX_quafu,
            "RY": self.get_RY_quafu,
            "RZ": self.get_RZ_quafu,
            "Rot": self.get_Rot_quafu,
            "PhaseShift": self.get_PhaseShift_quafu,
            "ControlledPhaseShift": self.get_ControlledPhaseShift_quafu,
            "CRX": self.get_CRX_quafu,
            "CRY": self.get_CRY_quafu,
            "CRZ": self.get_CRZ_quafu,
        }

        user = User()
        user.save_apitoken("gl5X6QkgJS5n6728FhtAEr07FqpdHumdM8WmJzrrMHA.9JzMwUzM3czN2EjOiAHelJCL1cDN6ICZpJye.9JiN1IzUIJiOicGbhJCLiQ1VKJiOiAXe0Jye")
        from quafu import Task
        self._task = Task()
        self._task.load_account()
        self._task.config(backend="ScQ-P10", shots=self._num_shots, compile=True, priority=2)



        # TODO: need to make sure each qubit only has one measurement! probs measurement measure selected qubits or all qubits

        self._rotations = []
        for measurement in self._measurements:
            if measurement.return_type is Expectation:
                if isinstance(measurement.obs, list):
                    for ob in measurement.obs:
                        self._rotations.extend(ob.diagonalizing_gates())
                else:
                    self._rotations.extend(measurement.obs.diagonalizing_gates())

            elif measurement.return_type is Probability:
                pass

            else:
                raise ValueError(f'hardware backend do not support {measurement.return_type} measurement!')


    def _update_parameters(self, *params):
        #print(len(params))
        '''
        Update the parameters of trainable gate according to new input parameters. The new input parameters are passed to this function when the `compiledCircuit` is called.

        Args:
            params (array): Array of the new parameters.

        **Example**

        .. code-block:: python3

            def circuitDef(*params):
                qai.RX(params[0], qubits=[1])
                qai.Hadamard(qubits=[0])
                qai.CNOT(qubits=[0,1])
                qai.RY(params[1], qubits=[0])
                return qai.expval(qai.PauliZ(qubits=[0]))
            #compile the quantum circuit
            circuit = qai.Circuit(circuitDef, 2, 0.54, 0.12)
            my_compilecircuit = circuit.compilecircuit(backend="jax")
    
        Compiled circuit will called this function and evaluate the circuit based on the updated parameters:

        >>> my_compilecircuit(0.54, 0.12)
        [DeviceArray(-9.4627275e-09, dtype=float32)]

        '''

        # print("_update_parameters: ", params, len(params))
        count = 0
        # print(self._operatorparams, len(self._operatorparams))

        for idx, trnblepars in self._trainableparams.items():
            len_tp = len(trnblepars)
            # print(len_tp)
            # print(count)
            # print(len(params))
            # print(params)
            # print(idx, self._operatorparams[idx],trnblepars)

            if len_tp > 0:
                #for i in range(len_tp):# trnblepars is list of indices of trainable parameters of the gate
                for i in range(len_tp):
                    pos = trnblepars[i]
                    # print(self._operatorparams[idx])
                    # print(self._operatorparams[idx][i])
                    # print(params[count+i].requires_grad)

                    # print(idx, pos, count, i, len(params), len(self._operatorparams[idx]))

                    self._operatorparams[idx][pos] = params[count + i]
                count = count + len_tp
                #print("count:  ", count)
        if len(params) != count:
            raise ValueError(f'Error!!!! number of parameters are not matched!! required {count} but {len(params)} are given')


    def _parser_circuit(self):
        '''
        Get tensor, name and parameters of each gate from the `Circuit` for ``Wave function vector method``.
        '''


        for idx, trnblepars in self._trainableparams.items():

            len_tp = len(trnblepars)

            if len_tp > 0:
                name = self._operatorname[idx]
                parameters = self._operatorparams[idx]
                parameters = [par.item() for par in parameters]
                qubits = self._appliedqubits[idx]
                self._gate_to_quafu[name](qubits, parameters)

            else:
                name = self._operatorname[idx]
                qubits = self._appliedqubits[idx]
                self._gate_to_quafu[name](qubits)

        for operator in self._rotations:
            name = operator.name
            #print(name)
            qubits = operator.qubits
            self._gate_to_quafu[name](qubits)

        measures = range(self._num_qubits)
        cbits = range(self._num_qubits)
        self._qc.measure(measures,  cbits=cbits)




    def real_execute(self, *params):
        r'''
        '''
        if len(params) == 0:
            pass
            
        else:
            self._update_parameters(*params)


        self._qc = quafu_QuantumCircuit(self._num_qubits)

        self._parser_circuit()

        #res = self._task.send(self._qc)
        from quafu import simulate
        res = simulate(self._qc, output="amplitudes")

        amplitudes = res.amplitudes
        #print(res)
        #print(amplitudes)


        # self._probabilities = np.zeros(2**self._num_qubits)
        # for i in range(2**self._num_qubits):
        #     str_bin_i = to_str_bin(i, self._num_qubits)
        #     p = amplitudes.get(str_bin_i, 0)
        #     #print(p)
        #     self._probabilities[i] = p
        # #print(self._probabilities)

        self._probabilities = amplitudes

        self._probabilities = self._probabilities.reshape([2 for _ in range(self._num_qubits)])

        probabilities = torch.from_numpy(self._probabilities)
        results = self.get_measurement_results(probabilities)

        if len(results) == 1:
            return results[0]

        #output must be a torch.tensor form
        try:
            results = torch.stack(results, 0)
        except:
            raise ValueError(f'You can not have multiple measurements with different shapes!!')

        return results


    def get_I_quafu(self, qubits):
        '''
        Get corresponding qiskit object.
        '''

        pass

    
    def get_Hadamard_quafu(self, qubits):
        '''
        Get corresponding qiskit object.
        '''
        qubit = qubits[0]
        self._qc.h(qubit)

    
    def get_PauliX_quafu(self, qubits):
        '''
        Get corresponding qiskit object.
        '''
        qubit = qubits[0]
        self._qc.x(qubit)

    
    def get_PauliY_quafu(self, qubits):
        '''
        Get corresponding qiskit object.
        '''
        qubit = qubits[0]
        self._qc.y(qubit)

    
    def get_PauliZ_quafu(self, qubits):
        '''
        Get corresponding qiskit object.
        '''
        qubit = qubits[0]
        self._qc.z(qubit)

    
    def get_S_quafu(self, qubits):
        '''
        Get corresponding qiskit object.
        '''
        qubit = qubits[0]
        self._qc.s(qubit)

    
    def get_T_quafu(self, qubits):
        '''
        Get corresponding qiskit object.
        '''
        qubit = qubits[0]
        self._qc.t(qubit)

    
    def get_SX_quafu(self, qubits):
        '''
        Get corresponding qiskit object.
        '''
        qubit = qubits[0]
        self._qc.sx(qubit)

    
    def get_CNOT_quafu(self, qubits):
        '''
        Get corresponding qiskit object.
        '''
        qubit = qubits[0]
        self._qc.cnot(qubit)

    
    def get_CZ_quafu(self, qubits):
        '''
        Get corresponding qiskit object.
        '''
        qubit = qubits[0]
        self._qc.cz(qubit)

    
    def get_CY_quafu(self, qubits):
        '''
        Get corresponding qiskit object.
        '''
        qubit = qubits[0]
        self._qc.cy(qubit)

    
    def get_SWAP_quafu(self, qubits):
        '''
        Get corresponding qiskit object.
        '''
        qubit_0 = qubits[0]
        qubit_1 = qubits[1]
        self._qc.swap(qubit_0, qubit_1)

    
    def get_CSWAP_quafu(self, qubits):
        '''
        Get corresponding qiskit object.
        '''

        qubit_0 = qubits[0]
        qubit_1 = qubits[1]
        self._qc.swap(qubit_0, qubit_1)

    
    def get_Toffoli_quafu(self, qubits):
        '''
        Get corresponding qiskit object.
        '''

        qubit_0 = qubits[0]
        qubit_1 = qubits[1]
        self._qc.toffoli(qubit_0, qubit_1)
        

    
    def get_RX_quafu(self, qubits, params):
        '''
        Get corresponding qiskit object.
        '''

        qubit = qubits[0]
        param = params[0]
        self._qc.rx(qubit, param)

    
    def get_RY_quafu(self, qubits, params):
        '''
        Get corresponding qiskit object.
        '''

        qubit = qubits[0]
        param = params[0]
        self._qc.ry(qubit, param)

    
    def get_RZ_quafu(self, qubits, params):
        '''
        Get corresponding qiskit object.
        '''
        qubit = qubits[0]
        param = params[0]
        self._qc.rz(qubit, param)

    
    def get_Rot_quafu(self, qubits):
        '''
        Get corresponding qiskit object.
        '''

        raise NotImplementedError

    
    def get_PhaseShift_quafu(self, qubits):
        '''
        Get corresponding qiskit object.
        '''

        qubit = qubits[0]
        self._qc.p(qubit)

    
    def get_ControlledPhaseShift_quafu(self, qubits):
        '''
        Get ControlledPhaseShift tensor
        '''

        raise NotImplementedError

    
    def get_CRX_quafu(self, qubits):
        '''
        Get CRX tensor
        '''

        raise NotImplementedError

    
    def get_CRY_quafu(self, qubits):
        '''
        Get CRY tensor
        '''

        raise NotImplementedError

    
    def get_CRZ_quafu(self, qubits):
        '''
        Get corresponding qiskit object.
        '''

        raise NotImplementedError