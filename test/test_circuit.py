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

import tedq as qai
from tedq.QInterpreter.operators.ops_abc import GateBase
from tedq.QInterpreter.operators.measurement import QuantumMeasurement
from tedq.QInterpreter.circuits.storage_base import StorageBase, CircuitStorage
from tedq.QInterpreter.circuits.circuit import Circuit
from tedq.quantum_error import QuantumStorageError

class Test_CircuitStorage():

    def test_construction(self):
        r"""
        """
        circuit_storage = CircuitStorage()
        assert isinstance(circuit_storage, StorageBase)
        assert hasattr(circuit_storage,"__enter__")
        assert hasattr(circuit_storage,"__exit__")

        print("Test CircuitStorage construction ok!")

    def test_recording(self):
        r"""
        """
        with CircuitStorage() as dummy_storage:
            assert CircuitStorage.recording()
            assert dummy_storage is CircuitStorage.active_context()
        assert dummy_storage.recording() is False

        print("Test CircuitStorage recording function ok!")

    def test_append(self):
        r"""
        """
        def append_obj(obj):
            CircuitStorage.append(obj)

        dummy_storage = CircuitStorage()
        assert len(dummy_storage.storage_context) == 0
        with dummy_storage:
            a = 1
            b = ["test", "ok"]
            c = ("circuit", "storage")
            d = {"tedq":"good", "paddle":"soso"}
            append_obj(a)
            append_obj(b)
            append_obj(c)
            append_obj(d)

        assert len(dummy_storage.storage_context) == 4
        assert dummy_storage.storage_context[0] is a
        assert dummy_storage.storage_context[1] is b
        assert dummy_storage.storage_context[2] is c
        assert dummy_storage.storage_context[3] is d

        print("Test CircuitStorage append function ok!")


    def test_remove(self):
        r"""
        """
        def append_obj(obj):
            CircuitStorage.append(obj)

        def remove_obj(obj):
            CircuitStorage.remove(obj)

        dummy_storage = CircuitStorage()
        with dummy_storage:
            a = 1
            b = ["test", "ok"]
            c = ("circuit", "storage")
            d = {"tedq":"good", "paddle":"soso"}
            append_obj(a)
            append_obj(b)
            append_obj(c)
            append_obj(d)

            assert a in dummy_storage.storage_context
            remove_obj(a)
            assert a not in dummy_storage.storage_context

            assert b in dummy_storage.storage_context
            remove_obj(b)
            assert b not in dummy_storage.storage_context

            assert c in dummy_storage.storage_context
            remove_obj(c)
            assert c not in dummy_storage.storage_context

            assert d in dummy_storage.storage_context
            remove_obj(d)
            assert d not in dummy_storage.storage_context

        print("Test CircuitStorage remove function ok!")


class test_Circuit:
    def test_construction(self):
        r"""
        """
        params = tuple([random.random() for _ in range(2)])
        def circuitDef(*params):
            qai.RY(params[0], qubits=[0])
            qai.RZ(params[1], qubits=[1])
            return qai.expval(qai.PauliZ(qubits=[0]))

        circuit = qai.Circuit(circuitDef, 2, *params)
        assert isinstance(circuit, Circuit)

        print("Test Quantum Circuit construction ok!")

    def test_incorrect_num_qubits(self):
        r"""
        """
        params = tuple([random.random() for _ in range(2)])
        def circuitDef(*params):
            qai.RY(params[0], qubits=[0])
            qai.RZ(params[1], qubits=[1])
            return qai.expval(qai.PauliZ(qubits=[0]))

        with pytest.raises(ValueError):
            circuit = qai.Circuit(circuitDef, 1, *params)

        print("Test incorrect # of circuit qubits ok!")


    def test_unknow_content_in_queue(self):
        r"""
        """
        def append_obj(obj):
            CircuitStorage.append(obj)
        params = tuple([random.random() for _ in range(2)])
        def circuitDef(*params):
            qai.RY(params[0], qubits=[0])
            qai.RZ(params[1], qubits=[1])
            a = "wrong"
            append_obj(a)
            return qai.expval(qai.PauliZ(qubits=[0]))

        with pytest.raises(QuantumStorageError):
            circuit = qai.Circuit(circuitDef, 1, *params)

        print("Test incorrect # of circuit qubits ok!")

    def test_gate_inside_circuit(self):
        r"""
        Test that a gate class can be instantiated INside quantum circuit,
        qubits is important here and do_queue is set to be True by default.
        """
        def circuitDef(*parameters):
            qai.Hadamard(qubits=[0]) 
            qai.PauliX(qubits=[1]) 
            qai.PauliY(qubits=[2]) 
            qai.PauliZ(qubits=[3]) 
            qai.I(qubits=[4]) 
            qai.S(qubits=[5]) 
            qai.T(qubits=[6]) 
            qai.SX(qubits=[7]) 
            qai.CNOT(qubits=[0, 1]) 
            qai.CY(qubits=[1, 2]) 
            qai.CZ(qubits=[2, 3]) 
            qai.SWAP(qubits=[3, 4]) 
            qai.CSWAP(qubits=[1, 2, 0])         
            qai.Toffoli(qubits=[1, 2, 0]) 
            qai.RX(parameters[0], qubits=[0]) 
            qai.RY(parameters[1], qubits=[1]) 
            qai.RZ(parameters[2], qubits=[2])         
            qai.Rot(parameters[3], parameters[4], parameters[5], qubits=[0]) 
            qai.PhaseShift(parameters[6], qubits=[0]) 
            qai.ControlledPhaseShift(parameters[7], qubits=[1, 0]) 
            qai.CRX(parameters[8], qubits=[0, 3]) 
            qai.CRY(parameters[9], qubits=[0, 3]) 
            qai.CRZ(parameters[10], qubits=[0, 3])
            return qai.expval(qai.PauliZ(qubits=[0]))

        params = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.11, 0,12)
        num_qubits = 8
        circuit = qai.Circuit(circuitDef, num_qubits, *params)

        gates = circuit.operators
        assert len(gates) == 23
        assert gates[0].name == "Hadamard"
        assert gates[22].name == "CRZ"

        for i in range(len(gates)):

            gate = gates[i]
            assert isinstance(gate, GateBase)

            current_id = gate.instance_id
            if i == 0:
                pre_id = current_id
            else:
                delta_id = current_id - pre_id
                pre_id = current_id
                assert delta_id == 1

        print("test gate construction inside circuit ok!")


    def test_measurement_inside_circuit(self):
        r"""
        Test that a gate class can be instantiated INside quantum circuit,
        qubits is important here and do_queue is set to be True by default.
        """
        def circuitDef(*parameters):
            qai.Toffoli(qubits=[1, 2, 0]) 
            qai.RX(parameters[0], qubits=[0]) 
            return [qai.expval(qai.PauliZ(qubits=[0])), qai.state(), qai.probs(qubits=[1])]

        params = (0.1,)
        num_qubits = 3
        circuit = qai.Circuit(circuitDef, num_qubits, *params)

        measurements = circuit.measurements
        assert len(measurements) == 3

        for meas in measurements:
        	assert isinstance(meas, QuantumMeasurement)

        print("test measurement construction inside circuit ok!")


class test_Circuit_io:
    r'''
    '''
    def test_reading_qsim_file(self):
        r'''
        '''
        qasm_circuit = qai.FromQasmFile("test.qsim")

        n_qubits = qasm_circuit.n_qubits
        assert n_qubits == 3

        def circuitDef():
            qasm_circuit()
            return qai.expval(qai.PauliZ(qubits=[0]))

        circuit = qai.Circuit(circuitDef, n_qubits)
        assert circuit.operators[2].qubits == [2]

        print("test reading quantum circuit from qsim file ok!")