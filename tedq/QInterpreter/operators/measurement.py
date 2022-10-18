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

# pylint: disable=line-too-long, trailing-whitespace, too-many-lines, too-many-instance-attributes, too-few-public-methods

r"""
This module contains the available measurement processes supported by QInterpreter 
(expectation, probability, state).
"""

from enum import Enum
from tedq.global_variables import GlobalVariables
from tedq.quantum_error import QuantumCircuitError, QuantumValueError
from .ops_abc import ObservableBase



class QuantumMeasurement:
    r"""
    A measurement process at the end of quantum circuit.
    Three types of measurement (expectation, probability, state) are available 
    by QInterpreter.

    Args:
        return_type (.MeasurementReturnTypes): Which type of measurement is going to apply, available types are: ``Expectation``, ``Probability`` and ``State``.
        obs (.ObservableBase): Optional, default is ``None``. The observable that is to be measured.
        qubits (int): Optional, default is ``None``. The qubit(s) that the measurement applied to. If ``obs`` is not ``None``, this can not be specified.

    """
    def __init__(self, return_type, obs=None, qubits=None, do_queue=True):
    
        self.return_type = return_type
        self.obs = obs
        self.do_queue = do_queue

        if qubits is not None and obs is not None:
            raise ValueError("If an observable is provied, the qubit(s) can not be specified!")

        self.qubits = qubits

        # Put this measurement into the circuit.
        if self.do_queue:
            self.circuit_queue()

    def circuit_queue(self):
        r"""
        Append the measurement into the end of a quantum circuit.
        """
        if self.obs is not None:
            # Obtain the current quantum circuit warehouse.
            _active_warehouse = GlobalVariables.get_value("global_deque")
            if bool(_active_warehouse):
                
                # multiple qubits expectation value measurement.
                if isinstance(self.obs, list):
                    # count from the last one
                    for ob in reversed(self.obs):
                        # Obtain the last ``Operator`` of current quantum circuit.
                        last_content = _active_warehouse[-1].storage_context[-1]
                        # The last operator should be the ``ob``, replace it
                        # withe the measurement process.
                        if last_content.instance_id == ob.instance_id:
                            _active_warehouse[-1].remove(ob)
                        else:
                            raise ValueError("Last content operator should be the same as 'obs'!")
                    # append the measurement into the queue
                    _active_warehouse[-1].append(self)                       



                # single qubit expectation value measurement.
                else:
                    # Obtain the last ``Operator`` of current quantum circuit.
                    last_content = _active_warehouse[-1].storage_context[-1]
                    # The last operator should be the ``obs``, replace it
                    # withe the measurement process.
                    if last_content.instance_id == self.obs.instance_id:
                        _active_warehouse[-1].remove(self.obs)
                        _active_warehouse[-1].append(self)
                    else:
                        raise ValueError("Last content operator should be the same as 'obs'!")
            else:
                raise QuantumCircuitError("No active warehouse, tedq software is problematic, need to check 'global_variables.py'")

        else:
            _active_warehouse = GlobalVariables.get_value("global_deque")
            if bool(_active_warehouse):
                # Obtain current quantum circuit and append this measurement to the end.
                _active_warehouse[-1]._append(self)  # pylint: disable=protected-access
            else:
                raise QuantumCircuitError("No active warehouse, tedq software is problematic, need to check 'global_variables.py'")


def expval(observable, do_queue=True):
    r"""Compute expectation value measurement result of the input observable.
    
    Args:
        observable (ObservableBase): An observable class which to be measured.

    Raises:
        QuantumValueError: `observable` is not an `ObservableBase` class.


    **Example**

    .. code-block:: python3

        def circuitDef(*params):
            qai.RX(params[0], qubits=[1])
            qai.Hadamard(qubits=[0])
            qai.CNOT(qubits=[0,1])
            qai.RY(params[1], qubits=[0])
            return qai.expval(qai.PauliZ(qubits=[0]))
            # for ZZ measurement
            #return qai.expval([qai.PauliZ(qubits=[0]), qai.PauliZ(qubits=[1])])
        #compile the quantum circuit
        circuit = qai.Circuit(circuitDef, 2, 0.54, 0.12)
        my_compilecircuit = circuit.compilecircuit(backend="jax")
    
    Execute the circuit and obtain the result:

    >>> my_compilecircuit(0.54, 0.12)
    [DeviceArray(-9.4627275e-09, dtype=float32)]

    
    """
    # multiple qubits expectation value measurement.
    if isinstance(observable, list):
        for ob in observable:
            if not isinstance(ob, ObservableBase):
                raise QuantumValueError(
                    f'{ob.name} is not a subclass of ObservableBase: cannot be used with expval'
                )


    # single qubit expectation value measurement.
    else:
        if not isinstance(observable, ObservableBase):
            raise QuantumValueError(
                f'{observable.name} is not a subclass of ObservableBase: cannot be used with expval'
            )

    return QuantumMeasurement(Expectation, obs=observable, do_queue=do_queue)


def var(observable, do_queue=True):
    r"""
    Obtain the variance of measurement for the supplied observable.
    """
    # Will be implemented later.
    raise NotImplementedError


def sample(observable, num_shots, do_queue=True):
    r"""
    Obtain the sample of measurement for the supplied observable.
    """
    # Will be implemented later.
    raise NotImplementedError


def probs(qubits=None, do_queue=True):
    r"""
    Compute the probability of each computational basis state

    //TODO: some description and fix the example
    

    **Example**

    .. code-block:: python3

        def circuitDef(*params):
            qai.RX(params[0], qubits=[1])
            qai.Hadamard(qubits=[0])
            qai.CNOT(qubits=[0,1])
            qai.RY(params[1], qubits=[0])
            return qai.probs(qai.PauliZ(qubits=[0]))
        #compile the quantum circuit
        circuit = qai.Circuit(circuitDef, 2, 0.54, 0.12)
        my_compilecircuit = circuit.compilecircuit(backend="jax")
    
    Execute the circuit and obtain the result:

    >>> my_compilecircuit(0.54, 0.12)
    [DeviceArray(-9.4627275e-09, dtype=float32)]

    The returned array is in lexicographic order, so corresponds
    to a :math:`50\%` chance of measuring either :math:`|00\rangle`
    or :math:`|01\rangle`.

    Args:
        wires (Sequence[int] or int): the wire the operation acts on
    """
    return QuantumMeasurement(Probability, qubits=qubits, do_queue=do_queue)


def state(do_queue=True):
    r"""Quantum state in the computational basis.


    //TODO: some description and fix the example
    

    **Example**

    .. code-block:: python3

        def circuitDef(*params):
            qai.RX(params[0], qubits=[1])
            qai.Hadamard(qubits=[0])
            qai.CNOT(qubits=[0,1])
            qai.RY(params[1], qubits=[0])
            return qai.probs(qai.PauliZ(qubits=[0]))
        #compile the quantum circuit
        circuit = qai.Circuit(circuitDef, 2, 0.54, 0.12)
        my_compilecircuit = circuit.compilecircuit(backend="jax")
    
    Execute the circuit and obtain the result:

    >>> my_compilecircuit(0.54, 0.12)
    [DeviceArray(-9.4627275e-09, dtype=float32)]

    """
    return QuantumMeasurement(State, do_queue=do_queue)


class MeasurementReturnTypes(Enum):
    r"""
    The class define the Enums for the different return types of measurement.
    """

    # pylint: disable=invalid-name

    Sample = "sample"
    Variance = "var"
    Expectation = "expval"
    Probability = "probs"
    State = "state"

    def __repr__(self):
        """String representation of the return types."""
        return str(self.value)


Sample = MeasurementReturnTypes.Sample

Variance = MeasurementReturnTypes.Variance

Expectation = MeasurementReturnTypes.Expectation

Probability = MeasurementReturnTypes.Probability

State = MeasurementReturnTypes.State
