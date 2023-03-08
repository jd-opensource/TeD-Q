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
This module contains the :class:`Circuit` class for converting user input quantum function
 into quantum circuit.
"""

# pylint: disable=line-too-long, trailing-whitespace, too-many-lines, too-many-instance-attributes, too-few-public-methods

from tedq.backends import PyTorchBackend, JaxBackend, QUDIOBackend, HardwareBackend_qiskit, HardwareBackend_quafu
from tedq.QInterpreter.operators.ops_abc import GateBase
from tedq.QInterpreter.operators.measurement import QuantumMeasurement
from tedq.QInterpreter.operators.measurement import probs as Probs
from tedq.quantum_error import QuantumStorageError, QuantumCircuitError
from .storage_base import CircuitStorage

from collections import defaultdict

import numpy as np
import torch
import itertools



class Circuit:
    r"""Circuit
    Converting user-define quantum function into tedq quantum circuit
    """
    def __init__(self, func, num_qubits, *params, **kwargs):
        if isinstance(func, dict):
            self._circuit = None
            self._operators = func['operators']
            self._measurements = func['measurements']
            self._init_state = func['init_state']
            self._num_qubits = num_qubits

        else:
            self._func = func
            if num_qubits:
                self._num_qubits = num_qubits
            else:
                raise ValueError("Error in Circuit class, num_qubits cannot be None!")

            parameter_shapes = kwargs.get('parameter_shapes', None)
            if parameter_shapes:
                #print(parameter_shapes)
                params = generate_parameters(parameter_shapes)
            with CircuitStorage() as circuit:
                self._func(*params)

            self._circuit = circuit

            # TODO: add checking, make sure state preparation at begin, measurement at end
            if self._circuit.storage_context[0]._is_preparation:
                self._init_state = self._circuit.storage_context[0]
                #print(self._init_state._is_preparation)
            else:
                self._init_state = None


            self._operators = [
                ops for ops in self._circuit.storage_context if isinstance(ops, GateBase)
            ]

            self._measurements = [
                ops for ops in self._circuit.storage_context if isinstance(ops, QuantumMeasurement)
            ]

            if not (
                len(self._operators) + len(self._measurements) + bool(self._init_state)
                == len(circuit.storage_context)
            ):
                #print(len(self._operators), len(self._measurements), self._init_state)
                raise QuantumStorageError("Error in Circuit class, unknown content in the storage_context!")

        if len(self._measurements) == 0:
            raise QuantumCircuitError("No measurement! please specify a quantum measurement!")

        maximum_qubit_number = self.maximum_qubit_num()
        #print(maximum_qubit_number, self._num_qubits)
        if maximum_qubit_number+1 > self._num_qubits: # since qubits in operator count from 0, so it need to be added 1
            raise ValueError(
                f'Input number of qubits is not large enough! '
                f'Maximum qubit number of operators is {maximum_qubit_number+1}'
                )

    # TODO: 放到一個另外的獨立文件
    def cycabc(self, numarg):
        r'''
        the fuck
        numarg: List of integer
        '''
        # lateral layer of each qubit
        q_layers = list(0 for i in range(self._num_qubits))

        # vertical layer for natural gradient descent
        layers = defaultdict(list) # default值以一個list()方法產生

        obs_list = defaultdict(list)
        factors_list = defaultdict(list)
        # Only the first one is useful
        num_params = defaultdict(list)

        num = 0
        for op in self._operators:


            tp = op.trainable_params
            len_tp = len(tp)


            op_qubits = op.qubits

            # update layer id
            new_layer_id = max([q_layers[i] for i in op_qubits])



            # one parameter operator
            if len_tp == 1:
                # parameter need to be used for natural gradient
                if num in numarg:
                    gen = op.generator
                    s = gen[0]
                    obs = gen[1]

                    obs_list[new_layer_id].append(obs)
                    factors_list[new_layer_id].append(s)
                    # Only the first one is useful
                    num_params[new_layer_id].append(num)

                    # update layer id
                    new_layer_id = new_layer_id + 1

            # quantum gate with more than one parameters can not use quantum natural gradient!
            if len_tp > 1:
                tmpt_num = range(num, num+len_tp)
                num_in_numarg = [n in numarg for n in tmpt_num]
                if any(num_in_numarg):
                    raise ValueError("quantum gate with more than one parameters can not use quantum natural gradient!")

            # update layer id
            for i in op_qubits:
                q_layers[i] = new_layer_id

            layers[new_layer_id].append(op)


            num = num + len_tp

        return (layers, obs_list, factors_list, num_params)

    def gen_circuits(self, numarg):
        r'''
        '''

        circuits_list = []

        (layers, obs_list, factors_list, num_params) = self.cycabc(numarg)

        for layer_id, obs in obs_list.items():
            operators = []
            qubits = []
            measurements = []

            num_param = num_params[layer_id][0]
            factors = factors_list[layer_id]

            for i in range(layer_id+1):
                operators.extend(layers[i])
            for ob in obs:
                # rotations
                operators.extend(ob.diagonalizing_gates())
                qubits.extend(ob.qubits)

            # TODO: use qubits=qubits, but need to change marginal_prob code
            # since qubits=qubits will destroy some wires, and then the
            # marginal_prob will not be corrected
            measurements.append(Probs(qubits=None, do_queue=False))#qubits=qubits

            quantum_circuit_dict = {'operators':operators, 
            'measurements':measurements, 'init_state':self._init_state}

            #print(operators)
            circuit = Circuit(quantum_circuit_dict, self._num_qubits)

            circuits_list.append((circuit, obs, factors, num_param))


        return circuits_list





    def maximum_qubit_num(self):
        r"""
        return maximum qubit number.
        """
        max_qubit = 0
        for ops in self.operators:
            bigger_qubit = max(ops.qubits)
            if bigger_qubit > max_qubit:
                max_qubit = bigger_qubit

        for ops in self.measurements:
            if ops.qubits is not None:
                bigger_qubit = max(ops.qubits)
                if bigger_qubit > max_qubit:
                    max_qubit = bigger_qubit
            if ops.obs is not None:
                # multiple qubits expectation value measurement.
                if isinstance(ops.obs, list):
                    for ob in ops.obs:
                        bigger_qubit = max(ob.qubits)
                        if bigger_qubit > max_qubit:
                            max_qubit = bigger_qubit                        

                # single qubit expectation value measurement.
                else:
                    bigger_qubit = max(ops.obs.qubits)
                    if bigger_qubit > max_qubit:
                        max_qubit = bigger_qubit

        return max_qubit

    @property
    def operators(self):
        r"""
        return operators inside the quantum circuit.
        """
        return self._operators

    @property
    def num_qubits(self):
        r"""
        return number of qubits of this circuit.
        """
        return self._num_qubits

    @property
    def measurements(self):
        r"""
        return measurement objects of this circuit.
        """
        return self._measurements

    @property
    def init_state(self):
        r"""
        return init_state objects of this circuit.
        """
        return self._init_state

    def compilecircuit(self, backend=None, **kwargs):
        r"""
        According to the user-specified input, compile this circuit to executable compiled circuit.
        """
        if backend == "jax":  #pylint: disable=no-else-return
            return JaxBackend(backend, self, **kwargs)
        elif backend == "pytorch":
            return PyTorchBackend(backend, self, **kwargs)
        elif backend == "pytorch_QUDIO":
            return QUDIOBackend(backend, self, **kwargs)
        elif backend == "IBMQ_hardware":
            return HardwareBackend_qiskit(backend, self, **kwargs)
        elif backend == "Quafu_hardware":
            return HardwareBackend_quafu(backend, self, **kwargs)
        else:
            raise ValueError(
                f'{backend}: unknown backend input'
            )
    def __str__(self):
        string = ""
        string+="# operators\n"
        for gate in self.operators:
            string += "qai."+str(gate.name)+"(qubits="+str(gate.qubits)+")"+"\n"
        string+="\n\n# measurements\n"
        for measurement in self.measurements:
            string += "qai.measurement.expval(qai."+str(measurement.obs.name)+"(qubits="+str(measurement.obs.qubits)+"))"+"\n"
        
        return string

    def metric_tensor(self, circuits_list, *parameters):  

        gs = []
        for (circuit, obs, factors, num_param) in circuits_list:
            #print(num_param)
            compiled_circuit = circuit.compilecircuit(backend="pytorch")
            probs = compiled_circuit(*parameters[:num_param])   

            scale = np.outer(factors, factors)
            scale = torch.from_numpy(scale)
            #print(scale, probs) 

            
            #print(obs)
            g = scale * cov_matrix(probs, obs)
            gs.append(g)
            #print("g: ",g)

        # create the block diagonal metric tensor
        return torch.block_diag(*(g for g in gs))

def cov_matrix(prob, obs, diag_approx=False):
    r'''
    '''
    prob = prob.type(torch.complex64)
    variances = []

    # diagonal variances
    #print(obs)
    for i, o in enumerate(obs):
        eigvals = o.eigvals
        eigvals = torch.from_numpy(eigvals)
        eigvals = eigvals.type(torch.complex64)
        qubits = o.qubits
        p = marginal_prob(prob, qubits)

        res = torch.dot(eigvals**2, p) - (torch.dot(eigvals, p)) ** 2
        variances.append(res)

    cov = torch.diag(torch.tensor(variances))

    if diag_approx:
        return cov

    for i, j in itertools.combinations(range(len(obs)), r=2):
        o1 = obs[i]
        o2 = obs[j]

        qubits_1 = o1.qubits
        qubits_2 = o2.qubits

        shared_qubits = list(qubits_1)
        shared_qubits.extend(qubits_2)

        l1 = o1.eigvals
        l1 = torch.from_numpy(l1)
        l1 = l1.type(torch.complex64)
        l2 = o2.eigvals
        l2 = torch.from_numpy(l2)
        l2 = l2.type(torch.complex64)
        l12 = torch.kron(l1, l2)

        p1 = marginal_prob(prob, qubits_1)
        p2 = marginal_prob(prob, qubits_2)
        p12 = marginal_prob(prob, shared_qubits)

        res = torch.dot(l12, p12) - torch.dot(l1, p1) * torch.dot(l2, p2)

        cov[i, j] = cov[i, j] + res
        cov[j, i] = cov[j, i] + res

    #print("cov: ", cov)
    return cov


def marginal_prob(prob, axis):
    """Compute the marginal probability given a joint probability distribution expressed as a tensor.
    Each random variable corresponds to a dimension.
    If the distribution arises from a quantum circuit measured in computational basis, each dimension
    corresponds to a wire. For example, for a 2-qubit quantum circuit `prob[0, 1]` is the probability of measuring the
    first qubit in state 0 and the second in state 1.
    Args:
        prob (tensor_like): 1D tensor of probabilities. This tensor should of size
            ``(2**N,)`` for some integer value ``N``.
        axis (list[int]): the axis for which to calculate the marginal
            probability distribution
    Returns:
        tensor_like: the marginal probabilities, of
        size ``(2**len(axis),)``
    **Example**
    >>> x = tf.Variable([1, 0, 0, 1.], dtype=tf.float64) / np.sqrt(2)
    >>> marginal_prob(x, axis=[0, 1])
    <tf.Tensor: shape=(4,), dtype=float64, numpy=array([0.70710678, 0.        , 0.        , 0.70710678])>
    >>> marginal_prob(x, axis=[0])
    <tf.Tensor: shape=(2,), dtype=float64, numpy=array([0.70710678, 0.70710678])>
    """
    prob = prob.view(-1)
    num_wires = int(np.log2(len(prob)))

    if num_wires == len(axis):
        return prob

    inactive_wires = tuple(set(range(num_wires)) - set(axis))
    prob = torch.reshape(prob, [2] * num_wires)
    prob = torch.sum(prob, axis=inactive_wires)
    return torch.flatten(prob)



def generate_parameters(parameter_shapes):
    r'''
    Generate random parameters from input parameter shapes
    '''
    import numpy as np  # pylint: disable=import-outside-toplevel
    params = []
    for shape in parameter_shapes:
        param = (np.random.rand(*shape) +0.01) * np.e/1.77# random number between 0 and pi/2.
        params.append(param)
    params = tuple(params)
    return params


