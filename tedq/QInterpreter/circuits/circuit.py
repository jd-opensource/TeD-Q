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
from tedq.quantum_error import QuantumStorageError, QuantumCircuitError
from .storage_base import CircuitStorage


class Circuit:
    r"""
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
