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
Test file for storage and circuit.
"""
import numpy as np
import random
import pytest
import torch

import tedq as qai
from tedq.QInterpreter.operators.ops_abc import GateBase
from tedq.QInterpreter.operators.measurement import QuantumMeasurement
from tedq.QInterpreter.circuits.storage_base import StorageBase, CircuitStorage
from tedq.QInterpreter.circuits.circuit import Circuit
from tedq.quantum_error import QuantumStorageError
from tedq.backends.compiled_circuit import CompiledCircuit

class Test_compiled_circuit():
    r"""
    """
    def test_construction(self):
        params = tuple([random.random() for _ in range(2)])
        def circuitDef(*params):
            qai.RY(params[0], qubits=[0])
            qai.RZ(params[1], qubits=[1])
            return qai.expval(qai.PauliZ(qubits=[0]))

        circuit = qai.Circuit(circuitDef, 2, *params)
        compiled_circuit = circuit.compilecircuit(backend="jax")

        assert isinstance(compiled_circuit, CompiledCircuit)

        print("Test compiled_circuit construction ok!")

    def test_operators_measuremts(self):
        r"""
        """
        params = tuple([random.random() for _ in range(2)])
        def circuitDef(*params):
            qai.RY(params[0], qubits=[0])
            qai.RZ(params[1], qubits=[1])
            return qai.expval(qai.PauliZ(qubits=[0]))

        circuit = qai.Circuit(circuitDef, 2, *params)
        compiled_circuit = circuit.compilecircuit(backend="jax")

        operators_cc = compiled_circuit.operators
        operators_c = circuit.operators

        measurements_cc = compiled_circuit.measurements
        measurements_c = circuit.measurements

        assert operators_cc == operators_c
        assert measurements_cc == measurements_c

        print("Test compiled_circuit operators and measurement ok!")

    def test_confliction_order_finder(self):
        """
        """
        params = tuple([random.random() for _ in range(2)])
        def circuitDef(*params):
            qai.RY(params[0], qubits=[0])
            qai.RZ(params[1], qubits=[1])
            return qai.expval(qai.PauliZ(qubits=[0]))

        circuit = qai.Circuit(circuitDef, 2, *params)

        with pytest.raises(ValueError, match="Error!!!! can not use contengra, opt_einsum and cyc at the same time!"):
            compiled_circuit = circuit.compilecircuit(backend="jax",use_cotengra=True, use_jdopttn=True)

        print("Test confliction on two contraction order finder ok!")

    def test_SVPM_axes_list_and_perms_list(self):
        """
        state vector propagation mode
        """
        params = tuple([random.random() for _ in range(2)])
        def circuitDef(*params):
            qai.RY(params[0], qubits=[0])
            qai.RZ(params[1], qubits=[1])
            return qai.expval(qai.PauliZ(qubits=[0]))

        circuit = qai.Circuit(circuitDef, 2, *params)
        compiled_circuit = circuit.compilecircuit(backend="jax")
        tensordot_axeslist = compiled_circuit._axeslist
        assert tensordot_axeslist == [([1], [1]), ([1], [0])]
        tensordot_permutationlist = compiled_circuit._permutationlist
        assert tensordot_permutationlist == [[1, 0], [0, 1]]

        print("Test state vector propagation mode tensordot axeslist and permutationlist ok!")


    def test_circuit_to_tensor_network_convertion(self):
        """
        """
        params = tuple([random.random() for _ in range(2)])
        def circuitDef(*params):
            qai.RY(params[0], qubits=[0])
            qai.RZ(params[1], qubits=[1])
            return [qai.expval(qai.PauliZ(qubits=[0])), qai.state()]

        circuit = qai.Circuit(circuitDef, 2, *params)
        import cotengra as ctg
        compiled_circuit = circuit.compilecircuit(backend="jax", use_cotengra=ctg)

        operands = compiled_circuit._cotengra_operands

        measurement_0_operand = operands[0]
        input_indices_scr_0 = measurement_0_operand[0]
        output_indices_scr_0 = measurement_0_operand[1]
        size_dict_0 = measurement_0_operand[2]
        assert input_indices_scr_0 == [['a'], ['b'], ['a', 'c'], ['b', 'd'], ['c', 'e'], ['d', 'f'], ['e', 'g'], ['g'], ['f']]
        assert output_indices_scr_0 == []
        assert size_dict_0 == {'a': 2, 'b': 2, 'c': 2, 'd': 2, 'e': 2, 'f': 2, 'g': 2}

        operands = compiled_circuit._cotengra_operands
        measurement_1_operand = operands[1]
        input_indices_scr_1 = measurement_1_operand[0]
        output_indices_scr_1 = measurement_1_operand[1]
        size_dict_1 = measurement_1_operand[2]
        print(input_indices_scr_1)
        assert input_indices_scr_1 == [['a'], ['b'], ['a', 'c'], ['b', 'd'], ['c'], ['d']]
        assert output_indices_scr_1 == ['c', 'd']
        assert size_dict_1 == {'a': 2, 'b': 2, 'c': 2, 'd': 2}

        print("Test circuit to tensor network convertion ok!")

    def test_trainable_parameters(self):
        """
        """
        params = tuple([random.random() for _ in range(4)])
        def circuitDef(*params):
            qai.Rot(params[0], params[1], params[2], qubits=[0], trainable_params=[0, 2])
            qai.RZ(params[3], qubits=[1])
            return qai.expval(qai.PauliZ(qubits=[0]))

        circuit = qai.Circuit(circuitDef, 2, *params)
        compiled_circuit = circuit.compilecircuit(backend="jax")# here can be jax or pytorch backend

        gate_parameters = compiled_circuit.gate_parameters
        assert gate_parameters == [[params[0], params[1], params[2]], [params[3]]]

        all_params = compiled_circuit.parameters
        assert all_params == [params[0], params[1], params[2], params[3]]

        trainable_params = compiled_circuit.trainable_parameters
        assert trainable_params == [params[0], params[2], params[3]]

        print("Test trainable parameters ok!")

    def test_update_parameters(self):
        """
        """
        params = tuple([torch.tensor([random.random()], requires_grad = True) for _ in range(4)])
        def circuitDef(*params):
            qai.Rot(params[0], params[1], params[2], qubits=[0], trainable_params=[0, 2])
            qai.RZ(params[3], qubits=[1])
            return qai.expval(qai.PauliZ(qubits=[0]))

        circuit = qai.Circuit(circuitDef, 2, *params)
        compiled_circuit = circuit.compilecircuit(backend="pytorch")# must use pytorch backend because of jax jit function on execute function

        gate_parameters = compiled_circuit.gate_parameters
        assert gate_parameters == [[params[0], params[1], params[2]], [params[3]]]

        all_params = compiled_circuit.parameters
        assert all_params == [params[0], params[1], params[2], params[3]]

        trainable_params = compiled_circuit.trainable_parameters
        assert trainable_params == [params[0], params[2], params[3]]

        new_params = tuple([torch.tensor([random.random()], requires_grad = True) for _ in range(3)])
        compiled_circuit(*new_params)# must use new_params, not updated_params!!

        updated_params = (new_params[0], params[1], new_params[1], new_params[2])

        gate_parameters = compiled_circuit.gate_parameters
        #print(updated_params)
        #print(gate_parameters)
        a_list = [[gate_parameters[0][0].item(), gate_parameters[0][1].item(), gate_parameters[0][2].item()],[gate_parameters[1][0].item()]]
        #print(a_list)
        b_list = [[updated_params[0].item(), updated_params[1].item(), updated_params[2].item()], [updated_params[3].item()]]
        #print(b_list)
        assert a_list == b_list

        all_params = compiled_circuit.parameters
        a_list = [all_params[0].item(), all_params[1].item(), all_params[2].item(), all_params[3].item()]
        b_list = [updated_params[0].item(), updated_params[1].item(), updated_params[2].item(), updated_params[3].item()]
        assert a_list == b_list

        trainable_params = compiled_circuit.trainable_parameters
        a_list = [trainable_params[0].item(), trainable_params[1].item(), trainable_params[2].item()]
        b_list = [new_params[0].item(), new_params[1].item(), new_params[2].item()]
        assert a_list == b_list

        print("Test update parameters ok!")