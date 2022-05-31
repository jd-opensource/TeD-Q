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
Test file for gate class construction and basic function.
"""

import itertools
import functools
import pytest
import numpy as np
from numpy.linalg import multi_dot
import random

import tedq as qai
from tedq.QInterpreter.operators.ops_abc import GateBase, ObservableBase
from tedq.QInterpreter.operators.measurement import QuantumMeasurement, Expectation, Probability, State


# pylint: disable=no-self-use, no-member, protected-access, pointless-statement


class TestGateOperators:
    """Test gate class construction."""
    gate_operators_list = [
        "Hadamard",
        "PauliX",
        "PauliY",
        "PauliZ",
        "I",
        "S",
        "T",
        "SX",
        "CNOT",
        "CZ",
        "CY",
        "SWAP",
        "CSWAP",
        "Toffoli",
        "RX",
        "RY",
        "RZ",
        "PhaseShift",
        "ControlledPhaseShift",
        "CRX",
        "CRY",
        "CRZ",
    ]

    def test_construction(self):
        r"""
        Test that a gate class can be instantiated OUTside quantum circuit,
        qubits and do_queue are not important here.
        """
        Hadamard = qai.Hadamard(qubits=[0], do_queue = False)
        assert isinstance(Hadamard, GateBase)

        PauliX = qai.PauliX(qubits=[1], do_queue = False)
        assert isinstance(PauliX, GateBase)

        PauliY = qai.PauliY(qubits=[2], do_queue = False)
        assert isinstance(PauliY, GateBase)

        PauliZ = qai.PauliZ(qubits=[3], do_queue = False)
        assert isinstance(PauliZ, GateBase)

        I = qai.I(qubits=[4], do_queue = False)
        assert isinstance(I, GateBase)

        S = qai.S(qubits=[5], do_queue = False)
        assert isinstance(S, GateBase)

        T = qai.T(qubits=[6], do_queue = False)
        assert isinstance(T, GateBase)

        SX = qai.SX(qubits=[7], do_queue = False)
        assert isinstance(SX, GateBase)

        CNOT = qai.CNOT(qubits=[0, 1], do_queue = False)
        assert isinstance(CNOT, GateBase)

        CY = qai.CY(qubits=[1, 2], do_queue = False)
        assert isinstance(CY, GateBase)

        CZ = qai.CZ(qubits=[2, 3], do_queue = False)
        assert isinstance(CZ, GateBase)

        SWAP = qai.SWAP(qubits=[3, 4], do_queue = False)
        assert isinstance(SWAP, GateBase)

        CSWAP = qai.CSWAP(qubits=[1, 2, 0], do_queue = False)
        assert isinstance(CSWAP, GateBase)
        
        Toffoli = qai.Toffoli(qubits=[1, 2, 0], do_queue = False)
        assert isinstance(Toffoli, GateBase)

        RX = qai.RX(0.5, qubits=[0], do_queue = False)
        assert isinstance(RX, GateBase)

        RY = qai.RY(0.5, qubits=[1], do_queue = False)
        assert isinstance(RY, GateBase)

        RZ = qai.RZ(0.5, qubits=[2], do_queue = False)
        assert isinstance(RZ, GateBase)
        
        Rot = qai.Rot(0.3, 0.4, 0.5, qubits=[0], do_queue = False)
        assert isinstance(Rot, GateBase)

        PhaseShift = qai.PhaseShift(0.5, qubits=[0], do_queue = False)
        assert isinstance(PhaseShift, GateBase)

        ControlledPhaseShift = qai.ControlledPhaseShift(0.5, qubits=[1, 0], do_queue = False)
        assert isinstance(ControlledPhaseShift, GateBase)

        CRX = qai.CRX(0.5, qubits=[0, 1], do_queue = False)
        assert isinstance(CRX, GateBase)

        CRY = qai.CRY(0.5, qubits=[0, 1], do_queue = False)
        assert isinstance(CRY, GateBase)

        CRZ = qai.CRZ(0.5, qubits=[0, 1], do_queue = False)
        assert isinstance(CRZ, GateBase)

        print("Test gate operators construction outside circuit ok!")

    def test_incorrect_num_qubits(self):
        r'''
        '''
        secure_random = random.SystemRandom()
        random_gate_name = "qai." + secure_random.choice(self.gate_operators_list)
        gate_operator_class = eval(random_gate_name)
        num_qubits = getattr(gate_operator_class, "_num_qubits", None)
        assert num_qubits is not None
        qubits = [secure_random.randint(0,100) for _ in range(num_qubits+1)]
        num_params = getattr(gate_operator_class, "_num_params", None)
        params = tuple([secure_random.random() for _ in range(num_params)])

        with pytest.raises(ValueError):
            random_gate_operator = gate_operator_class(*params, qubits = qubits, do_queue = False)

        print("Test incorrect number of qubits ok!")

    def test_trainable_parameters(self):
        r'''
        '''
        Rot = qai.Rot(0.3, 0.4, 0.5, qubits=[0], do_queue = False)# default all parameters are trainable
        trainable_params_index = Rot.trainable_params
        assert trainable_params_index == [0, 1, 2]

        Rot = qai.Rot(0.3, 0.4, 0.5, qubits=[0], do_queue = False, trainable_params=[1])
        trainable_params_index = Rot.trainable_params
        assert trainable_params_index == [1]

        print("Test trainable patameters ok!")


    def test_incorrect_num_params(self):
        r'''
        '''
        secure_random = random.SystemRandom()
        random_gate_name = "qai." + secure_random.choice(self.gate_operators_list)
        gate_operator_class = eval(random_gate_name)
        num_qubits = getattr(gate_operator_class, "_num_qubits", None)
        qubits = [secure_random.randint(0,100) for _ in range(num_qubits)]
        num_params = getattr(gate_operator_class, "_num_params", None)
        assert num_qubits is not None
        params = tuple([secure_random.random() for _ in range(num_params+1)])

        with pytest.raises(ValueError):
            random_gate_operator = gate_operator_class(*params, qubits = qubits, do_queue = False)

        print("Test incorrect number of parameters ok!")


    def test_no_qubits_args_passed(self):
        r'''
        '''
        secure_random = random.SystemRandom()
        random_gate_name = "qai." + secure_random.choice(self.gate_operators_list)
        gate_operator_class = eval(random_gate_name)
        num_params = getattr(gate_operator_class, "_num_params", None)
        params = tuple([secure_random.random() for _ in range(num_params)])

        with pytest.raises(TypeError):
            random_gate_operator = gate_operator_class(*params, do_queue = False)

        print("Test no qubits argument passed ok!")

    def test_decomposition(self):
        r"""
        """
        # Will be implemented later.
        raise NotImplementedError

    def test_adjoint(self):
        r"""
        """
        Hadamard = qai.Hadamard(qubits=[0], do_queue = False)
        adjoint_ops = Hadamard.adjoint()
        matrix_mul = np.matmul(Hadamard.matrix, adjoint_ops.matrix)
        I = np.identity(2**Hadamard.num_qubits)
        delta_matrix = matrix_mul - I
        assert (delta_matrix < 1.0e-3).all()

        PauliX = qai.PauliX(qubits=[1], do_queue = False)
        adjoint_ops = PauliX.adjoint()
        matrix_mul = np.matmul(PauliX.matrix, adjoint_ops.matrix)
        I = np.identity(2**PauliX.num_qubits)
        delta_matrix = matrix_mul - I
        assert (delta_matrix < 1.0e-3).all()

        PauliY = qai.PauliY(qubits=[2], do_queue = False)
        adjoint_ops = PauliY.adjoint()
        matrix_mul = np.matmul(PauliY.matrix, adjoint_ops.matrix)
        I = np.identity(2**PauliY.num_qubits)
        delta_matrix = matrix_mul - I
        assert (delta_matrix < 1.0e-3).all()

        PauliZ = qai.PauliZ(qubits=[3], do_queue = False)
        adjoint_ops = PauliZ.adjoint()
        matrix_mul = np.matmul(PauliZ.matrix, adjoint_ops.matrix)
        I = np.identity(2**PauliZ.num_qubits)
        delta_matrix = matrix_mul - I
        assert (delta_matrix < 1.0e-3).all()

        I = qai.I(qubits=[4], do_queue = False)
        adjoint_ops = I.adjoint()
        matrix_mul = np.matmul(I.matrix, adjoint_ops.matrix)
        I = np.identity(2**I.num_qubits)
        delta_matrix = matrix_mul - I
        assert (delta_matrix < 1.0e-3).all()

        S = qai.S(qubits=[5], do_queue = False)
        adjoint_ops = S.adjoint()
        matrix_mul = np.matmul(S.matrix, adjoint_ops.matrix)
        I = np.identity(2**S.num_qubits)
        delta_matrix = matrix_mul - I
        assert (delta_matrix < 1.0e-3).all()

        T = qai.T(qubits=[6], do_queue = False)
        adjoint_ops = T.adjoint()
        matrix_mul = np.matmul(T.matrix, adjoint_ops.matrix)
        I = np.identity(2**T.num_qubits)
        delta_matrix = matrix_mul - I
        assert (delta_matrix < 1.0e-3).all()

        SX = qai.SX(qubits=[7], do_queue = False)
        adjoint_ops = SX.adjoint()
        matrix_mul = np.matmul(SX.matrix, adjoint_ops.matrix)
        I = np.identity(2**SX.num_qubits)
        delta_matrix = matrix_mul - I
        assert (delta_matrix < 1.0e-3).all()

        CNOT = qai.CNOT(qubits=[0, 1], do_queue = False)
        adjoint_ops = CNOT.adjoint()
        matrix_mul = np.matmul(CNOT.matrix, adjoint_ops.matrix)
        I = np.identity(2**CNOT.num_qubits)
        delta_matrix = matrix_mul - I
        assert (delta_matrix < 1.0e-3).all()

        CY = qai.CY(qubits=[1, 2], do_queue = False)
        adjoint_ops = CY.adjoint()
        matrix_mul = np.matmul(CY.matrix, adjoint_ops.matrix)
        I = np.identity(2**CY.num_qubits)
        delta_matrix = matrix_mul - I
        assert (delta_matrix < 1.0e-3).all()

        CZ = qai.CZ(qubits=[2, 3], do_queue = False)
        adjoint_ops = CZ.adjoint()
        matrix_mul = np.matmul(CZ.matrix, adjoint_ops.matrix)
        I = np.identity(2**CZ.num_qubits)
        delta_matrix = matrix_mul - I
        assert (delta_matrix < 1.0e-3).all()

        SWAP = qai.SWAP(qubits=[3, 4], do_queue = False)
        adjoint_ops = SWAP.adjoint()
        matrix_mul = np.matmul(SWAP.matrix, adjoint_ops.matrix)
        I = np.identity(2**SWAP.num_qubits)
        delta_matrix = matrix_mul - I
        assert (delta_matrix < 1.0e-3).all()

        CSWAP = qai.CSWAP(qubits=[1, 2, 0], do_queue = False)
        adjoint_ops = CSWAP.adjoint()
        matrix_mul = np.matmul(CSWAP.matrix, adjoint_ops.matrix)
        I = np.identity(2**CSWAP.num_qubits)
        delta_matrix = matrix_mul - I
        assert (delta_matrix < 1.0e-3).all()

        Toffoli = qai.Toffoli(qubits=[1, 2, 0], do_queue = False)
        adjoint_ops = Toffoli.adjoint()
        matrix_mul = np.matmul(Toffoli.matrix, adjoint_ops.matrix)
        I = np.identity(2**Toffoli.num_qubits)
        delta_matrix = matrix_mul - I
        assert (delta_matrix < 1.0e-3).all()

        RX = qai.RX(0.5, qubits=[0], do_queue = False)
        adjoint_ops = RX.adjoint()
        matrix_mul = np.matmul(RX.matrix, adjoint_ops.matrix)
        I = np.identity(2**RX.num_qubits)
        delta_matrix = matrix_mul - I
        assert (delta_matrix < 1.0e-3).all()

        RY = qai.RY(0.5, qubits=[1], do_queue = False)
        adjoint_ops = RY.adjoint()
        matrix_mul = np.matmul(RY.matrix, adjoint_ops.matrix)
        I = np.identity(2**RY.num_qubits)
        delta_matrix = matrix_mul - I
        assert (delta_matrix < 1.0e-3).all()

        RZ = qai.RZ(0.5, qubits=[2], do_queue = False)
        adjoint_ops = RZ.adjoint()
        matrix_mul = np.matmul(RZ.matrix, adjoint_ops.matrix)
        I = np.identity(2**RZ.num_qubits)
        delta_matrix = matrix_mul - I
        assert (delta_matrix < 1.0e-3).all()

        Rot = qai.Rot(0.3, 0.4, 0.5, qubits=[0], do_queue = False)
        adjoint_ops = Rot.adjoint()
        matrix_mul = np.matmul(Rot.matrix, adjoint_ops.matrix)
        I = np.identity(2**Rot.num_qubits)
        delta_matrix = matrix_mul - I
        assert (delta_matrix < 1.0e-3).all()

        PhaseShift = qai.PhaseShift(0.5, qubits=[0], do_queue = False)
        adjoint_ops = PhaseShift.adjoint()
        matrix_mul = np.matmul(PhaseShift.matrix, adjoint_ops.matrix)
        I = np.identity(2**PhaseShift.num_qubits)
        delta_matrix = matrix_mul - I
        assert (delta_matrix < 1.0e-3).all()

        ControlledPhaseShift = qai.ControlledPhaseShift(0.5, qubits=[1, 0], do_queue = False)
        adjoint_ops = ControlledPhaseShift.adjoint()
        matrix_mul = np.matmul(ControlledPhaseShift.matrix, adjoint_ops.matrix)
        I = np.identity(2**ControlledPhaseShift.num_qubits)
        delta_matrix = matrix_mul - I
        assert (delta_matrix < 1.0e-3).all()

        CRX = qai.CRX(0.5, qubits=[0, 1], do_queue = False)
        adjoint_ops = CRX.adjoint()
        matrix_mul = np.matmul(CRX.matrix, adjoint_ops.matrix)
        I = np.identity(2**CRX.num_qubits)
        delta_matrix = matrix_mul - I
        assert (delta_matrix < 1.0e-3).all()

        CRY = qai.CRY(0.5, qubits=[0, 1], do_queue = False)
        adjoint_ops = CRY.adjoint()
        matrix_mul = np.matmul(CRY.matrix, adjoint_ops.matrix)
        I = np.identity(2**CRY.num_qubits)
        delta_matrix = matrix_mul - I
        assert (delta_matrix < 1.0e-3).all()

        CRZ = qai.CRZ(0.5, qubits=[0, 1], do_queue = False)
        adjoint_ops = CRZ.adjoint()
        matrix_mul = np.matmul(CRZ.matrix, adjoint_ops.matrix)
        I = np.identity(2**CRZ.num_qubits)
        delta_matrix = matrix_mul - I
        assert (delta_matrix < 1.0e-3).all()

        print("Test adjoint ok!")

    def test_inversion(self):
        r"""
        """
        Hadamard = qai.Hadamard(qubits=[0], do_queue = False)
        origin_name = Hadamard.name
        inversed_ops = Hadamard.inverse()
        inversed_name = origin_name + ".inv"
        assert inversed_ops.name == inversed_name
        inversed_matrix = inversed_ops.matrix
        re_inversed_ops = inversed_ops.inverse()
        assert re_inversed_ops.name == origin_name
        matrix_mul = np.matmul(re_inversed_ops.matrix, inversed_matrix)
        I = np.identity(2**Hadamard.num_qubits)
        delta_matrix = matrix_mul - I
        assert (delta_matrix < 1.0e-3).all()


        PauliX = qai.PauliX(qubits=[1], do_queue = False)
        origin_name = PauliX.name
        inversed_ops = PauliX.inverse()
        inversed_name = origin_name + ".inv"
        assert inversed_ops.name == inversed_name
        inversed_matrix = inversed_ops.matrix
        re_inversed_ops = inversed_ops.inverse()
        assert re_inversed_ops.name == origin_name
        matrix_mul = np.matmul(re_inversed_ops.matrix, inversed_matrix)
        I = np.identity(2**PauliX.num_qubits)
        delta_matrix = matrix_mul - I
        assert (delta_matrix < 1.0e-3).all()

        PauliY = qai.PauliY(qubits=[2], do_queue = False)
        origin_name = PauliY.name
        inversed_ops = PauliY.inverse()
        inversed_name = origin_name + ".inv"
        assert inversed_ops.name == inversed_name
        inversed_matrix = inversed_ops.matrix
        re_inversed_ops = inversed_ops.inverse()
        assert re_inversed_ops.name == origin_name
        matrix_mul = np.matmul(re_inversed_ops.matrix, inversed_matrix)
        I = np.identity(2**PauliY.num_qubits)
        delta_matrix = matrix_mul - I
        assert (delta_matrix < 1.0e-3).all()

        PauliZ = qai.PauliZ(qubits=[3], do_queue = False)
        origin_name = PauliZ.name
        inversed_ops = PauliZ.inverse()
        inversed_name = origin_name + ".inv"
        assert inversed_ops.name == inversed_name
        inversed_matrix = inversed_ops.matrix
        re_inversed_ops = inversed_ops.inverse()
        assert re_inversed_ops.name == origin_name
        matrix_mul = np.matmul(re_inversed_ops.matrix, inversed_matrix)
        I = np.identity(2**PauliZ.num_qubits)
        delta_matrix = matrix_mul - I
        assert (delta_matrix < 1.0e-3).all()

        I = qai.I(qubits=[4], do_queue = False)
        origin_name = I.name
        inversed_ops = I.inverse()
        inversed_name = origin_name + ".inv"
        assert inversed_ops.name == inversed_name
        inversed_matrix = inversed_ops.matrix
        re_inversed_ops = inversed_ops.inverse()
        assert re_inversed_ops.name == origin_name
        matrix_mul = np.matmul(re_inversed_ops.matrix, inversed_matrix)
        I = np.identity(2**I.num_qubits)
        delta_matrix = matrix_mul - I
        assert (delta_matrix < 1.0e-3).all()

        S = qai.S(qubits=[5], do_queue = False)
        origin_name = S.name
        inversed_ops = S.inverse()
        inversed_name = origin_name + ".inv"
        assert inversed_ops.name == inversed_name
        inversed_matrix = inversed_ops.matrix
        re_inversed_ops = inversed_ops.inverse()
        assert re_inversed_ops.name == origin_name
        matrix_mul = np.matmul(re_inversed_ops.matrix, inversed_matrix)
        I = np.identity(2**S.num_qubits)
        delta_matrix = matrix_mul - I
        assert (delta_matrix < 1.0e-3).all()

        T = qai.T(qubits=[6], do_queue = False)
        origin_name = T.name
        inversed_ops = T.inverse()
        inversed_name = origin_name + ".inv"
        assert inversed_ops.name == inversed_name
        inversed_matrix = inversed_ops.matrix
        re_inversed_ops = inversed_ops.inverse()
        assert re_inversed_ops.name == origin_name
        matrix_mul = np.matmul(re_inversed_ops.matrix, inversed_matrix)
        I = np.identity(2**T.num_qubits)
        delta_matrix = matrix_mul - I
        assert (delta_matrix < 1.0e-3).all()

        SX = qai.SX(qubits=[7], do_queue = False)
        origin_name = SX.name
        inversed_ops = SX.inverse()
        inversed_name = origin_name + ".inv"
        assert inversed_ops.name == inversed_name
        inversed_matrix = inversed_ops.matrix
        re_inversed_ops = inversed_ops.inverse()
        assert re_inversed_ops.name == origin_name
        matrix_mul = np.matmul(re_inversed_ops.matrix, inversed_matrix)
        I = np.identity(2**SX.num_qubits)
        delta_matrix = matrix_mul - I
        assert (delta_matrix < 1.0e-3).all()

        CNOT = qai.CNOT(qubits=[0, 1], do_queue = False)
        origin_name = CNOT.name
        inversed_ops = CNOT.inverse()
        inversed_name = origin_name + ".inv"
        assert inversed_ops.name == inversed_name
        inversed_matrix = inversed_ops.matrix
        re_inversed_ops = inversed_ops.inverse()
        assert re_inversed_ops.name == origin_name
        matrix_mul = np.matmul(re_inversed_ops.matrix, inversed_matrix)
        I = np.identity(2**CNOT.num_qubits)
        delta_matrix = matrix_mul - I
        assert (delta_matrix < 1.0e-3).all()

        CY = qai.CY(qubits=[1, 2], do_queue = False)
        origin_name = CY.name
        inversed_ops = CY.inverse()
        inversed_name = origin_name + ".inv"
        assert inversed_ops.name == inversed_name
        inversed_matrix = inversed_ops.matrix
        re_inversed_ops = inversed_ops.inverse()
        assert re_inversed_ops.name == origin_name
        matrix_mul = np.matmul(re_inversed_ops.matrix, inversed_matrix)
        I = np.identity(2**CY.num_qubits)
        delta_matrix = matrix_mul - I
        assert (delta_matrix < 1.0e-3).all()

        CZ = qai.CZ(qubits=[2, 3], do_queue = False)
        origin_name = CZ.name
        inversed_ops = CZ.inverse()
        inversed_name = origin_name + ".inv"
        assert inversed_ops.name == inversed_name
        inversed_matrix = inversed_ops.matrix
        re_inversed_ops = inversed_ops.inverse()
        assert re_inversed_ops.name == origin_name
        matrix_mul = np.matmul(re_inversed_ops.matrix, inversed_matrix)
        I = np.identity(2**CZ.num_qubits)
        delta_matrix = matrix_mul - I
        assert (delta_matrix < 1.0e-3).all()

        SWAP = qai.SWAP(qubits=[3, 4], do_queue = False)
        origin_name = SWAP.name
        inversed_ops = SWAP.inverse()
        inversed_name = origin_name + ".inv"
        assert inversed_ops.name == inversed_name
        inversed_matrix = inversed_ops.matrix
        re_inversed_ops = inversed_ops.inverse()
        assert re_inversed_ops.name == origin_name
        matrix_mul = np.matmul(re_inversed_ops.matrix, inversed_matrix)
        I = np.identity(2**SWAP.num_qubits)
        delta_matrix = matrix_mul - I
        assert (delta_matrix < 1.0e-3).all()

        CSWAP = qai.CSWAP(qubits=[1, 2, 0], do_queue = False)
        origin_name = CSWAP.name
        inversed_ops = CSWAP.inverse()
        inversed_name = origin_name + ".inv"
        assert inversed_ops.name == inversed_name
        inversed_matrix = inversed_ops.matrix
        re_inversed_ops = inversed_ops.inverse()
        assert re_inversed_ops.name == origin_name
        matrix_mul = np.matmul(re_inversed_ops.matrix, inversed_matrix)
        I = np.identity(2**CSWAP.num_qubits)
        delta_matrix = matrix_mul - I
        assert (delta_matrix < 1.0e-3).all()

        Toffoli = qai.Toffoli(qubits=[1, 2, 0], do_queue = False)
        origin_name = Toffoli.name
        inversed_ops = Toffoli.inverse()
        inversed_name = origin_name + ".inv"
        assert inversed_ops.name == inversed_name
        inversed_matrix = inversed_ops.matrix
        re_inversed_ops = inversed_ops.inverse()
        assert re_inversed_ops.name == origin_name
        matrix_mul = np.matmul(re_inversed_ops.matrix, inversed_matrix)
        I = np.identity(2**Toffoli.num_qubits)
        delta_matrix = matrix_mul - I
        assert (delta_matrix < 1.0e-3).all()

        RX = qai.RX(0.5, qubits=[0], do_queue = False)
        origin_name = RX.name
        inversed_ops = RX.inverse()
        inversed_name = origin_name + ".inv"
        assert inversed_ops.name == inversed_name
        inversed_matrix = inversed_ops.matrix
        re_inversed_ops = inversed_ops.inverse()
        assert re_inversed_ops.name == origin_name
        matrix_mul = np.matmul(re_inversed_ops.matrix, inversed_matrix)
        I = np.identity(2**RX.num_qubits)
        delta_matrix = matrix_mul - I
        assert (delta_matrix < 1.0e-3).all()

        RY = qai.RY(0.5, qubits=[1], do_queue = False)
        origin_name = RY.name
        inversed_ops = RY.inverse()
        inversed_name = origin_name + ".inv"
        assert inversed_ops.name == inversed_name
        inversed_matrix = inversed_ops.matrix
        re_inversed_ops = inversed_ops.inverse()
        assert re_inversed_ops.name == origin_name
        matrix_mul = np.matmul(re_inversed_ops.matrix, inversed_matrix)
        I = np.identity(2**RY.num_qubits)
        delta_matrix = matrix_mul - I
        assert (delta_matrix < 1.0e-3).all()

        RZ = qai.RZ(0.5, qubits=[2], do_queue = False)
        origin_name = RZ.name
        inversed_ops = RZ.inverse()
        inversed_name = origin_name + ".inv"
        assert inversed_ops.name == inversed_name
        inversed_matrix = inversed_ops.matrix
        re_inversed_ops = inversed_ops.inverse()
        assert re_inversed_ops.name == origin_name
        matrix_mul = np.matmul(re_inversed_ops.matrix, inversed_matrix)
        I = np.identity(2**RZ.num_qubits)
        delta_matrix = matrix_mul - I
        assert (delta_matrix < 1.0e-3).all()

        Rot = qai.Rot(0.3, 0.4, 0.5, qubits=[0], do_queue = False)
        origin_name = Rot.name
        inversed_ops = Rot.inverse()
        inversed_name = origin_name + ".inv"
        assert inversed_ops.name == inversed_name
        inversed_matrix = inversed_ops.matrix
        re_inversed_ops = inversed_ops.inverse()
        assert re_inversed_ops.name == origin_name
        matrix_mul = np.matmul(re_inversed_ops.matrix, inversed_matrix)
        I = np.identity(2**Rot.num_qubits)
        delta_matrix = matrix_mul - I
        assert (delta_matrix < 1.0e-3).all()

        PhaseShift = qai.PhaseShift(0.5, qubits=[0], do_queue = False)
        origin_name = PhaseShift.name
        inversed_ops = PhaseShift.inverse()
        inversed_name = origin_name + ".inv"
        assert inversed_ops.name == inversed_name
        inversed_matrix = inversed_ops.matrix
        re_inversed_ops = inversed_ops.inverse()
        assert re_inversed_ops.name == origin_name
        matrix_mul = np.matmul(re_inversed_ops.matrix, inversed_matrix)
        I = np.identity(2**PhaseShift.num_qubits)
        delta_matrix = matrix_mul - I
        assert (delta_matrix < 1.0e-3).all()

        ControlledPhaseShift = qai.ControlledPhaseShift(0.5, qubits=[1, 0], do_queue = False)
        origin_name = ControlledPhaseShift.name
        inversed_ops = ControlledPhaseShift.inverse()
        inversed_name = origin_name + ".inv"
        assert inversed_ops.name == inversed_name
        inversed_matrix = inversed_ops.matrix
        re_inversed_ops = inversed_ops.inverse()
        assert re_inversed_ops.name == origin_name
        matrix_mul = np.matmul(re_inversed_ops.matrix, inversed_matrix)
        I = np.identity(2**ControlledPhaseShift.num_qubits)
        delta_matrix = matrix_mul - I
        assert (delta_matrix < 1.0e-3).all()

        CRX = qai.CRX(0.5, qubits=[0, 1], do_queue = False)
        origin_name = CRX.name
        inversed_ops = CRX.inverse()
        inversed_name = origin_name + ".inv"
        assert inversed_ops.name == inversed_name
        inversed_matrix = inversed_ops.matrix
        re_inversed_ops = inversed_ops.inverse()
        assert re_inversed_ops.name == origin_name
        matrix_mul = np.matmul(re_inversed_ops.matrix, inversed_matrix)
        I = np.identity(2**CRX.num_qubits)
        delta_matrix = matrix_mul - I
        assert (delta_matrix < 1.0e-3).all()

        CRY = qai.CRY(0.5, qubits=[0, 1], do_queue = False)
        origin_name = CRY.name
        inversed_ops = CRY.inverse()
        inversed_name = origin_name + ".inv"
        assert inversed_ops.name == inversed_name
        inversed_matrix = inversed_ops.matrix
        re_inversed_ops = inversed_ops.inverse()
        assert re_inversed_ops.name == origin_name
        matrix_mul = np.matmul(re_inversed_ops.matrix, inversed_matrix)
        I = np.identity(2**CRY.num_qubits)
        delta_matrix = matrix_mul - I
        assert (delta_matrix < 1.0e-3).all()

        CRZ = qai.CRZ(0.5, qubits=[0, 1], do_queue = False)
        origin_name = CRZ.name
        inversed_ops = CRZ.inverse()
        inversed_name = origin_name + ".inv"
        assert inversed_ops.name == inversed_name
        inversed_matrix = inversed_ops.matrix
        re_inversed_ops = inversed_ops.inverse()
        assert re_inversed_ops.name == origin_name
        matrix_mul = np.matmul(re_inversed_ops.matrix, inversed_matrix)
        I = np.identity(2**CRZ.num_qubits)
        delta_matrix = matrix_mul - I
        assert (delta_matrix < 1.0e-3).all()

        print("Test inversion ok!")


class TestObservableOperators:
    r'''
    '''
    obs = {"Hadamard", "PauliX", "PauliY", "PauliZ"}

    def diagonalizing_gates(self):
        r"""
        """
        # Will be implemented later.
        raise NotImplementedError

    def test_construction(self):
        r"""
        Test that a observable class can be instantiated outside quantum circuit,
        qubits and do_queue are not important here.
        """
        Hadamard = qai.Hadamard(qubits=[0], do_queue = False)
        assert isinstance(Hadamard, ObservableBase)

        PauliX = qai.PauliX(qubits=[0], do_queue = False)
        assert isinstance(Hadamard, ObservableBase)

        PauliY = qai.PauliY(qubits=[0], do_queue = False)
        assert isinstance(Hadamard, ObservableBase)

        PauliZ = qai.PauliZ(qubits=[0], do_queue = False)
        assert isinstance(Hadamard, ObservableBase)

        print("Test observable construction outside the circuit ok!")


class TestMeasurement:
    r'''
    '''

    def test_construction(self):
        r'''
        '''
        expval_measurement = qai.expval(qai.PauliX(qubits=[0], do_queue = False), do_queue = False)
        assert isinstance(expval_measurement, QuantumMeasurement)

        probs_measurement = qai.probs(do_queue = False)
        assert isinstance(probs_measurement, QuantumMeasurement)

        state_measurement = qai.state(do_queue = False)
        assert isinstance(state_measurement, QuantumMeasurement)

        print("Test measurement construction outside the circuit ok!")

    def test_return_type(self):
        r'''
        '''
        expval_measurement = qai.expval(qai.PauliX(qubits=[0], do_queue = False), do_queue = False)
        return_type = expval_measurement.return_type
        #print(return_type)
        assert return_type == Expectation

        probs_measurement = qai.probs(do_queue = False)
        return_type = probs_measurement.return_type
        assert return_type == Probability

        state_measurement = qai.state(do_queue = False)
        return_type = state_measurement.return_type
        assert return_type == State

        print("Test measurement return type outside the circuit ok!")  

    def test_wrong_qubits_obs_input(self):
        obs = qai.PauliX(qubits=[0], do_queue = False)
        with pytest.raises(ValueError):
            QuantumMeasurement("expval", obs=obs, qubits=[0], do_queue=False)

        print("Test wrong qubits and observable input ok!")

