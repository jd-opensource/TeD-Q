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


#TODO: add tests for cotengra and cyc_cotengra
r"""
Test file for storage and circuit.
"""
import numpy as np
import random
import pytest
from jax import numpy as jnp
import torch
import jax

import tedq as qai
from tedq.QInterpreter.operators.ops_abc import GateBase
from tedq.QInterpreter.operators.measurement import QuantumMeasurement
from tedq.QInterpreter.circuits.storage_base import StorageBase, CircuitStorage
from tedq.QInterpreter.circuits.circuit import Circuit
from tedq.quantum_error import QuantumStorageError
from tedq.backends.pytorch_backend import PyTorchBackend

class Test_pytorch_backend():
    r"""
    """
    def test_construction(self):
        params = tuple([random.random() for _ in range(2)])
        def circuitDef(*params):
            qai.RY(params[0], qubits=[0])
            qai.RZ(params[1], qubits=[1])
            return qai.expval(qai.PauliZ(qubits=[0]))

        circuit = qai.Circuit(circuitDef, 2, *params)

        compiled_circuit = circuit.compilecircuit(backend="pytorch")
        assert isinstance(compiled_circuit, PyTorchBackend)
        assert compiled_circuit.diff_method == "back_prop"
        assert compiled_circuit.backend == "pytorch"

        compiled_circuit = circuit.compilecircuit(backend="pytorch", diff_method = "param_shift")
        assert isinstance(compiled_circuit, PyTorchBackend)
        assert compiled_circuit.diff_method == "param_shift"
        assert compiled_circuit.backend == "pytorch"

        print("Test pytorch backend construction ok!")

    def test_unknown_interface(self):
        r"""
        """
        params = tuple([random.random() for _ in range(2)])
        def circuitDef(*params):
            qai.RY(params[0], qubits=[0])
            qai.RZ(params[1], qubits=[1])
            return qai.expval(qai.PauliZ(qubits=[0]))

        circuit = qai.Circuit(circuitDef, 2, *params)

        with pytest.raises(ValueError):
            compiled_circuit = circuit.compilecircuit(backend="pytorch", interface="jax")

        print("Test unknown interface ok!")

    def test_unknown_differential_method(self):
        r"""
        """
        params = tuple([random.random() for _ in range(2)])
        def circuitDef(*params):
            qai.RY(params[0], qubits=[0])
            qai.RZ(params[1], qubits=[1])
            return qai.expval(qai.PauliZ(qubits=[0]))

        circuit = qai.Circuit(circuitDef, 2, *params)

        with pytest.raises(ValueError):
            compiled_circuit = circuit.compilecircuit(backend="pytorch", diff_method = "arbitrary")

        print("Test unknown differential method ok!")

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
            compiled_circuit = circuit.compilecircuit(backend="pytorch",use_cotengra=True, use_jdopttn=True)

        print("Test confliction on two contraction order finder ok!")

    def test_diff_input_dtype_param_shift_method(self):
        def circuitDef(*params):
            qai.RX(params[0], qubits=[0])
            qai.RY(params[1], qubits=[0])
            return [qai.expval(qai.PauliZ(qubits=[0])), qai.expval(qai.PauliX(qubits=[1]))]

        num_qubits = 2
        _a = torch.tensor([0.54], dtype = torch.float32, requires_grad = True)
        _b = torch.tensor([0.12], dtype = torch.float32, requires_grad = True)
        params = (_a, _b)
        circuit = qai.Circuit(circuitDef, num_qubits, *params)
        my_compilecircuit = circuit.compilecircuit(backend="pytorch", diff_method="param_shift")
        results = my_compilecircuit(*params)

        _a = torch.tensor([0.54], dtype = torch.complex64, requires_grad = True)
        _b = torch.tensor([0.12], dtype = torch.complex64, requires_grad = True)
        params = (_a, _b)
        circuit = qai.Circuit(circuitDef, num_qubits, *params)
        my_compilecircuit = circuit.compilecircuit(backend="pytorch", diff_method="param_shift")
        results = my_compilecircuit(*params)

        _a = torch.tensor([0.54], dtype = torch.float32, requires_grad = True)
        _b = torch.tensor([0.12], dtype = torch.complex64, requires_grad = True)
        params = (_a, _b)
        circuit = qai.Circuit(circuitDef, num_qubits, *params)
        my_compilecircuit = circuit.compilecircuit(backend="pytorch", diff_method="param_shift")
        with pytest.raises(ValueError, match="input parameters for parameter shift method must have the same data type!"):
            results = my_compilecircuit(*params)

        print("test different input data types for parameter shift method")

    def test_tensor_data(self):
        """
        """
        Hadamard = qai.Hadamard(qubits=[0], do_queue = False)
        np_matrix_a = Hadamard.matrix
        np_matrix_b = np.asarray(np_matrix_a, dtype = np.complex64).reshape([2,2])
        np_matrix = np.round(np_matrix_b, decimals = 5)
        torch_matrix_a = PyTorchBackend.get_Hadamard_tensor(paramslist=[])
        torch_matrix_b = torch_matrix_a.detach().numpy()
        torch_matrix = np.round(torch_matrix_b, decimals = 5)
        assert (np_matrix == torch_matrix).all()

        PauliX = qai.PauliX(qubits=[1], do_queue = False)
        np_matrix_a = PauliX.matrix
        np_matrix_b = np.asarray(np_matrix_a, dtype = np.complex64).reshape([2,2])
        np_matrix = np.round(np_matrix_b, decimals = 5)
        torch_matrix_a = PyTorchBackend.get_PauliX_tensor(paramslist=[])
        torch_matrix_b = torch_matrix_a.detach().numpy()
        torch_matrix = np.round(torch_matrix_b, decimals = 5)
        assert (np_matrix == torch_matrix).all()

        PauliY = qai.PauliY(qubits=[2], do_queue = False)
        np_matrix_a = PauliY.matrix
        np_matrix_b = np.asarray(np_matrix_a, dtype = np.complex64).reshape([2,2])
        np_matrix = np.round(np_matrix_b, decimals = 5)
        torch_matrix_a = PyTorchBackend.get_PauliY_tensor(paramslist=[])
        torch_matrix_b = torch_matrix_a.detach().numpy()
        torch_matrix = np.round(torch_matrix_b, decimals = 5)
        assert (np_matrix == torch_matrix).all()

        PauliZ = qai.PauliZ(qubits=[3], do_queue = False)
        np_matrix_a = PauliZ.matrix
        np_matrix_b = np.asarray(np_matrix_a, dtype = np.complex64).reshape([2,2])
        np_matrix = np.round(np_matrix_b, decimals = 5)
        torch_matrix_a = PyTorchBackend.get_PauliZ_tensor(paramslist=[])
        torch_matrix_b = torch_matrix_a.detach().numpy()
        torch_matrix = np.round(torch_matrix_b, decimals = 5)
        assert (np_matrix == torch_matrix).all()

        I = qai.I(qubits=[4], do_queue = False)
        np_matrix_a = I.matrix
        np_matrix_b = np.asarray(np_matrix_a, dtype = np.complex64).reshape([2,2])
        np_matrix = np.round(np_matrix_b, decimals = 5)
        torch_matrix_a = PyTorchBackend.get_I_tensor(paramslist=[])
        torch_matrix_b = torch_matrix_a.detach().numpy()
        torch_matrix = np.round(torch_matrix_b, decimals = 5)
        assert (np_matrix == torch_matrix).all()

        S = qai.S(qubits=[5], do_queue = False)
        np_matrix_a = S.matrix
        np_matrix_b = np.asarray(np_matrix_a, dtype = np.complex64).reshape([2,2])
        np_matrix = np.round(np_matrix_b, decimals = 5)
        torch_matrix_a = PyTorchBackend.get_S_tensor(paramslist=[])
        torch_matrix_b = torch_matrix_a.detach().numpy()
        torch_matrix = np.round(torch_matrix_b, decimals = 5)
        assert (np_matrix == torch_matrix).all()

        T = qai.T(qubits=[6], do_queue = False)
        np_matrix_a = T.matrix
        np_matrix_b = np.asarray(np_matrix_a, dtype = np.complex64).reshape([2,2])
        np_matrix = np.round(np_matrix_b, decimals = 5)
        torch_matrix_a = PyTorchBackend.get_T_tensor(paramslist=[])
        torch_matrix_b = torch_matrix_a.detach().numpy()
        torch_matrix = np.round(torch_matrix_b, decimals = 5)
        assert (np_matrix == torch_matrix).all()

        SX = qai.SX(qubits=[7], do_queue = False)
        np_matrix_a = SX.matrix
        np_matrix_b = np.asarray(np_matrix_a, dtype = np.complex64).reshape([2,2])
        np_matrix = np.round(np_matrix_b, decimals = 5)
        torch_matrix_a = PyTorchBackend.get_SX_tensor(paramslist=[])
        torch_matrix_b = torch_matrix_a.detach().numpy()
        torch_matrix = np.round(torch_matrix_b, decimals = 5)
        #print(np_matrix)
        #print(torch_matrix)
        assert (np_matrix == torch_matrix).all()

        CNOT = qai.CNOT(qubits=[0, 1], do_queue = False)
        np_matrix_a = CNOT.matrix
        np_matrix_b = np.asarray(np_matrix_a, dtype = np.complex64).reshape([2,2,2,2])
        np_matrix = np.round(np_matrix_b, decimals = 5)
        torch_matrix_a = PyTorchBackend.get_CNOT_tensor(paramslist=[])
        torch_matrix_b = torch_matrix_a.detach().numpy()
        torch_matrix = np.round(torch_matrix_b, decimals = 5)
        assert (np_matrix == torch_matrix).all()

        CY = qai.CY(qubits=[1, 2], do_queue = False)
        np_matrix_a = CY.matrix
        np_matrix_b = np.asarray(np_matrix_a, dtype = np.complex64).reshape([2,2,2,2])
        np_matrix = np.round(np_matrix_b, decimals = 5)
        torch_matrix_a = PyTorchBackend.get_CY_tensor(paramslist=[])
        torch_matrix_b = torch_matrix_a.detach().numpy()
        torch_matrix = np.round(torch_matrix_b, decimals = 5)
        assert (np_matrix == torch_matrix).all()

        CZ = qai.CZ(qubits=[2, 3], do_queue = False)
        np_matrix_a = CZ.matrix
        np_matrix_b = np.asarray(np_matrix_a, dtype = np.complex64).reshape([2,2,2,2])
        np_matrix = np.round(np_matrix_b, decimals = 5)
        torch_matrix_a = PyTorchBackend.get_CZ_tensor(paramslist=[])
        torch_matrix_b = torch_matrix_a.detach().numpy()
        torch_matrix = np.round(torch_matrix_b, decimals = 5)
        assert (np_matrix == torch_matrix).all()

        SWAP = qai.SWAP(qubits=[3, 4], do_queue = False)
        np_matrix_a = SWAP.matrix
        np_matrix_b = np.asarray(np_matrix_a, dtype = np.complex64).reshape([2,2,2,2])
        np_matrix = np.round(np_matrix_b, decimals = 5)
        torch_matrix_a = PyTorchBackend.get_SWAP_tensor(paramslist=[])
        torch_matrix_b = torch_matrix_a.detach().numpy()
        torch_matrix = np.round(torch_matrix_b, decimals = 5)
        assert (np_matrix == torch_matrix).all()

        CSWAP = qai.CSWAP(qubits=[1, 2, 0], do_queue = False)
        np_matrix_a = CSWAP.matrix
        np_matrix_b = np.asarray(np_matrix_a, dtype = np.complex64).reshape([2,2,2,2,2,2])
        np_matrix = np.round(np_matrix_b, decimals = 5)
        torch_matrix_a = PyTorchBackend.get_CSWAP_tensor(paramslist=[])
        torch_matrix_b = torch_matrix_a.detach().numpy()
        torch_matrix = np.round(torch_matrix_b, decimals = 5)
        assert (np_matrix == torch_matrix).all()

        Toffoli = qai.Toffoli(qubits=[1, 2, 0], do_queue = False)
        np_matrix_a = Toffoli.matrix
        np_matrix_b = np.asarray(np_matrix_a, dtype = np.complex64).reshape([2,2,2,2,2,2])
        np_matrix = np.round(np_matrix_b, decimals = 5)
        torch_matrix_a = PyTorchBackend.get_Toffoli_tensor(paramslist=[])
        torch_matrix_b = torch_matrix_a.detach().numpy()
        torch_matrix = np.round(torch_matrix_b, decimals = 5)
        assert (np_matrix == torch_matrix).all()

        params = (torch.tensor([0.5], dtype=torch.float32),)
        RX = qai.RX(*params, qubits=[0], do_queue = False)
        np_matrix_a = RX.matrix
        np_matrix_b = np.asarray(np_matrix_a, dtype = np.complex64).reshape([2,2])
        np_matrix = np.round(np_matrix_b, decimals = 5)
        torch_matrix_a = PyTorchBackend.get_RX_tensor(paramslist=[*params])
        torch_matrix_b = torch_matrix_a.detach().numpy()
        torch_matrix = np.round(torch_matrix_b, decimals = 5)
        assert (np_matrix == torch_matrix).all()

        params = (torch.tensor([0.5], dtype=torch.float32),)
        RY = qai.RY(*params, qubits=[1], do_queue = False)
        np_matrix_a = RY.matrix
        np_matrix_b = np.asarray(np_matrix_a, dtype = np.complex64).reshape([2,2])
        np_matrix = np.round(np_matrix_b, decimals = 5)
        torch_matrix_a = PyTorchBackend.get_RY_tensor(paramslist=[*params])
        torch_matrix_b = torch_matrix_a.detach().numpy()
        torch_matrix = np.round(torch_matrix_b, decimals = 5)
        assert (np_matrix == torch_matrix).all()

        params = (torch.tensor([0.5], dtype=torch.float32),)
        RZ = qai.RZ(*params, qubits=[2], do_queue = False)
        np_matrix_a = RZ.matrix
        np_matrix_b = np.asarray(np_matrix_a, dtype = np.complex64).reshape([2,2])
        np_matrix = np.round(np_matrix_b, decimals = 5)
        torch_matrix_a = PyTorchBackend.get_RZ_tensor(paramslist=[*params])
        torch_matrix_b = torch_matrix_a.detach().numpy()
        torch_matrix = np.round(torch_matrix_b, decimals = 5)
        assert (np_matrix == torch_matrix).all()

        params = (torch.tensor([0.3], dtype=torch.float32), torch.tensor([0.4], dtype=torch.float32), torch.tensor([0.5], dtype=torch.float32))
        Rot = qai.Rot(*params, qubits=[0], do_queue = False)
        np_matrix_a = Rot.matrix
        np_matrix_b = np.asarray(np_matrix_a, dtype = np.complex64).reshape([2,2])
        np_matrix = np.round(np_matrix_b, decimals = 5)
        torch_matrix_a = PyTorchBackend.get_Rot_tensor(paramslist=[*params])
        torch_matrix_b = torch_matrix_a.detach().numpy()
        torch_matrix = np.round(torch_matrix_b, decimals = 5)
        #print(np_matrix)
        #print(torch_matrix)
        assert (np_matrix == torch_matrix).all()

        params = (torch.tensor([0.5], dtype=torch.float32),)
        PhaseShift = qai.PhaseShift(*params, qubits=[0], do_queue = False)
        np_matrix_a = PhaseShift.matrix
        np_matrix_b = np.asarray(np_matrix_a, dtype = np.complex64).reshape([2,2])
        np_matrix = np.round(np_matrix_b, decimals = 5)
        torch_matrix_a = PyTorchBackend.get_PhaseShift_tensor(paramslist=[*params])
        torch_matrix_b = torch_matrix_a.detach().numpy()
        torch_matrix = np.round(torch_matrix_b, decimals = 5)
        assert (np_matrix == torch_matrix).all()

        params = (torch.tensor([0.5], dtype=torch.float32),)
        ControlledPhaseShift = qai.ControlledPhaseShift(*params, qubits=[1, 0], do_queue = False)
        np_matrix_a = ControlledPhaseShift.matrix
        np_matrix_b = np.asarray(np_matrix_a, dtype = np.complex64).reshape([2,2,2,2])
        np_matrix = np.round(np_matrix_b, decimals = 5)
        torch_matrix_a = PyTorchBackend.get_ControlledPhaseShift_tensor(paramslist=[*params])
        torch_matrix_b = torch_matrix_a.detach().numpy()
        torch_matrix = np.round(torch_matrix_b, decimals = 5)
        assert (np_matrix == torch_matrix).all()

        params = (torch.tensor([0.5], dtype=torch.float32),)
        CRX = qai.CRX(*params, qubits=[0, 1], do_queue = False)
        np_matrix_a = CRX.matrix
        np_matrix_b = np.asarray(np_matrix_a, dtype = np.complex64).reshape([2,2,2,2])
        np_matrix = np.round(np_matrix_b, decimals = 5)
        torch_matrix_a = PyTorchBackend.get_CRX_tensor(paramslist=[*params])
        torch_matrix_b = torch_matrix_a.detach().numpy()
        torch_matrix = np.round(torch_matrix_b, decimals = 5)
        assert (np_matrix == torch_matrix).all()

        params = (torch.tensor([0.5], dtype=torch.float32),)
        CRY = qai.CRY(*params, qubits=[0, 1], do_queue = False)
        np_matrix_a = CRY.matrix
        np_matrix_b = np.asarray(np_matrix_a, dtype = np.complex64).reshape([2,2,2,2])
        np_matrix = np.round(np_matrix_b, decimals = 5)
        torch_matrix_a = PyTorchBackend.get_CRY_tensor(paramslist=[*params])
        torch_matrix_b = torch_matrix_a.detach().numpy()
        torch_matrix = np.round(torch_matrix_b, decimals = 5)
        assert (np_matrix == torch_matrix).all()

        params = (torch.tensor([0.5], dtype=torch.float32),)
        CRZ = qai.CRZ(*params, qubits=[0, 1], do_queue = False)
        np_matrix_a = CRZ.matrix
        np_matrix_b = np.asarray(np_matrix_a, dtype = np.complex64).reshape([2,2,2,2])
        np_matrix = np.round(np_matrix_b, decimals = 5)
        torch_matrix_a = PyTorchBackend.get_CRZ_tensor(paramslist=[*params])
        torch_matrix_b = torch_matrix_a.detach().numpy()
        torch_matrix = np.round(torch_matrix_b, decimals = 5)
        assert (np_matrix == torch_matrix).all()
        print("Test tensor data ok!")

    def test_init_state(self):
        """
        """
        def circuitDef(*params):
            return qai.state()

        num_qubits = random.randint(1, 10)
        circuit = qai.Circuit(circuitDef, num_qubits)
        my_compilecircuit = circuit.compilecircuit(backend="pytorch")
        results = my_compilecircuit()
        state = results[0]
        state = state.detach().numpy()
        state = np.round(state, decimals = 5)
        num_elements = 2 ** num_qubits
        array = [0. for _ in range(num_elements)]
        array[0] = 1.
        state_tensor = np.array(array, dtype=np.complex64)
        shape = [2 for _ in range(num_qubits)]
        state_tensor = state_tensor.reshape(shape)
        state_tensor = np.round(state_tensor, decimals = 5)
        assert (state == state_tensor).all()

        print("Test state initialization ok!")

    def test_SVPM_forward(self):
        """
        """
        #pytorch backend only support list of measurements of the same dimension,
        #since finally a torch.stack function is used to combine all the measurement result into a torch.tensor object
        #expectation value measurement test
        def circuitDef(*params):
            qai.RX(params[0], qubits=[0])
            qai.RY(params[1], qubits=[0])
            return [qai.expval(qai.PauliZ(qubits=[0])), qai.expval(qai.PauliX(qubits=[1]))]

        num_qubits = 2
        _a = torch.tensor([0.54], dtype = torch.float32, requires_grad = True)
        _b = torch.tensor([0.12], dtype = torch.float32, requires_grad = True)
        params = (_a, _b)
        circuit = qai.Circuit(circuitDef, num_qubits, *params)
        my_compilecircuit = circuit.compilecircuit(backend="pytorch")
        results = my_compilecircuit(*params)
        expval_result = results
        expval_result = expval_result.detach().numpy()
        expval_result = np.round(expval_result, decimals = 5)
        expval_correct = np.array([0.85154057, 0.0], dtype = jnp.float32)
        expval_correct = np.round(expval_correct, decimals = 5)
        assert (expval_result == expval_correct).all()

        #probs measurement test
        def circuitDef(*params):
            qai.RX(params[0], qubits=[0])
            qai.RY(params[1], qubits=[0])
            return qai.probs()

        num_qubits = 2
        _a = torch.tensor([0.54], dtype = torch.float32, requires_grad = True)
        _b = torch.tensor([0.12], dtype = torch.float32, requires_grad = True)
        params = (_a, _b)
        circuit = qai.Circuit(circuitDef, num_qubits, *params)
        my_compilecircuit = circuit.compilecircuit(backend="pytorch")
        results = my_compilecircuit(*params)
        expval_result = results
        expval_result = expval_result.detach().numpy()
        expval_result = np.squeeze(expval_result)
        expval_result = np.round(expval_result, decimals = 5)
        expval_correct = np.array([[0.9257702, 0.],[0.07422972, 0.]], dtype = jnp.float32)
        expval_correct = np.round(expval_correct, decimals = 5)
        assert (expval_result == expval_correct).all()

        #partial probs measurement test
        def circuitDef(*params):
            qai.RX(params[0], qubits=[0])
            qai.RY(params[1], qubits=[0])
            return qai.probs(qubits=[1])

        num_qubits = 2
        _a = torch.tensor([0.54], dtype = torch.float32, requires_grad = True)
        _b = torch.tensor([0.12], dtype = torch.float32, requires_grad = True)
        params = (_a, _b)
        circuit = qai.Circuit(circuitDef, num_qubits, *params)
        my_compilecircuit = circuit.compilecircuit(backend="pytorch")
        results = my_compilecircuit(*params)
        expval_result = results
        expval_result = expval_result.detach().numpy()
        expval_result = np.squeeze(expval_result)
        expval_result = np.round(expval_result, decimals = 5)
        expval_correct = np.array([1.0, 0.0], dtype = jnp.float32)
        expval_correct = np.round(expval_correct, decimals = 5)
        assert (expval_result == expval_correct).all() 

        #state measurement test
        def circuitDef(*params):
            qai.RX(params[0], qubits=[0])
            qai.RY(params[1], qubits=[0])
            return qai.state()

        num_qubits = 2
        _a = torch.tensor([0.54], dtype = torch.float32, requires_grad = True)
        _b = torch.tensor([0.12], dtype = torch.float32, requires_grad = True)
        params = (_a, _b)
        circuit = qai.Circuit(circuitDef, num_qubits, *params)
        my_compilecircuit = circuit.compilecircuit(backend="pytorch")
        results = my_compilecircuit(*params)
        expval_result = results
        expval_result = expval_result.detach().numpy()
        expval_result = np.squeeze(expval_result)
        expval_result = np.round(expval_result, decimals = 5)
        expval_correct = np.array([[0.9620366+0.01599429j, 0.+0.j],[0.05779156-0.26625147j, 0.+0.j]], dtype = np.complex64)
        expval_correct = np.round(expval_correct, decimals = 5)
        assert (expval_result == expval_correct).all()

        print("Test state vector propagation mode forward calculation ok!")


    #param_shift
    def test_SVPM_backward_back_prop(self):
        """
        pytorch backend only support list of measurements with the same dimensions.
        TODO: can add test of list of prbs or state measurements.
        """
        def circuitDef(*params):
            qai.RX(params[0], qubits=[0])
            qai.RY(params[1], qubits=[0])
            return [qai.expval(qai.PauliZ(qubits=[0])), qai.expval(qai.PauliX(qubits=[1]))]

        num_qubits = 2
        _a = torch.tensor([0.54], dtype = torch.float32, requires_grad = True)
        _b = torch.tensor([0.12], dtype = torch.float32, requires_grad = True)
        params = (_a, _b)
        circuit = qai.Circuit(circuitDef, num_qubits, *params)
        my_compilecircuit = circuit.compilecircuit(backend="pytorch")
        results = my_compilecircuit(*params)

        expval_result = results[0]
        expval_result.backward()
        result_grad_a = _a.grad.numpy()
        result_grad_b = _b.grad.numpy()
        result_grad_a = np.round(result_grad_a, decimals = 5)
        result_grad_b = np.round(result_grad_b, decimals = 5)
        correct_grad_a = np.array([-0.5104387], dtype=np.float32)
        correct_grad_b = np.array([-0.10267819], dtype=np.float32)
        correct_grad_a = np.round(correct_grad_a, decimals = 5)
        correct_grad_b = np.round(correct_grad_b, decimals = 5)
        assert result_grad_a == correct_grad_a
        assert result_grad_b == correct_grad_b

        # avoid backward through the graph more than one time
        # it is very important to have new input patameters,
        # otherwise the previous calculation, the grad inside the parameter tensor
        # will affect the new calculation result!
        _a = torch.tensor([0.54], dtype = torch.float32, requires_grad = True)
        _b = torch.tensor([0.12], dtype = torch.float32, requires_grad = True)
        params = (_a, _b)
        results = my_compilecircuit(*params)
        expval_result = results[1]
        expval_result.backward()
        result_grad_a = _a.grad.numpy()
        result_grad_b = _b.grad.numpy()
        result_grad_a = np.round(result_grad_a, decimals = 5)
        result_grad_b = np.round(result_grad_b, decimals = 5)
        correct_grad_a = np.array([0.], dtype=np.float32)
        correct_grad_b = np.array([0.], dtype=np.float32)
        correct_grad_a = np.round(correct_grad_a, decimals = 5)
        correct_grad_b = np.round(correct_grad_b, decimals = 5)
        assert result_grad_a == correct_grad_a
        assert result_grad_b == correct_grad_b

        print("Test state vector propagation mode backward calculation using pytorch built-in back propagation method ok!")

    def test_SVPM_backward_param_shift(self):
        """
        pytorch backend only support list of measurements with the same dimensions.
        TODO: can add test of list of prbs or state measurements.
        """
        def circuitDef(*params):
            qai.RX(params[0], qubits=[0])
            qai.RY(params[1], qubits=[0])
            return [qai.expval(qai.PauliZ(qubits=[0])), qai.expval(qai.PauliX(qubits=[1]))]

        num_qubits = 2
        _a = torch.tensor([0.54], dtype = torch.float32, requires_grad = True)
        _b = torch.tensor([0.12], dtype = torch.float32, requires_grad = True)
        params = (_a, _b)
        circuit = qai.Circuit(circuitDef, num_qubits, *params)
        my_compilecircuit = circuit.compilecircuit(backend="pytorch", diff_method="param_shift")
        results = my_compilecircuit(*params)

        expval_result = results[0]
        expval_result.backward()
        result_grad_a = _a.grad.numpy()
        result_grad_b = _b.grad.numpy()
        result_grad_a = np.round(result_grad_a, decimals = 5)
        result_grad_b = np.round(result_grad_b, decimals = 5)
        correct_grad_a = np.array([-0.5104387], dtype=np.float32)
        correct_grad_b = np.array([-0.10267819], dtype=np.float32)
        correct_grad_a = np.round(correct_grad_a, decimals = 5)
        correct_grad_b = np.round(correct_grad_b, decimals = 5)
        print(result_grad_a)
        print(correct_grad_a)
        assert result_grad_a == correct_grad_a
        assert result_grad_b == correct_grad_b

        # avoid backward through the graph more than one time
        # it is very important to have new input patameters,
        # otherwise the previous calculation, the grad inside the parameter tensor
        # will affect the new calculation result!
        _a = torch.tensor([0.54], dtype = torch.float32, requires_grad = True)
        _b = torch.tensor([0.12], dtype = torch.float32, requires_grad = True)
        params = (_a, _b)
        results = my_compilecircuit(*params)
        expval_result = results[1]
        expval_result.backward()
        result_grad_a = _a.grad.numpy()
        result_grad_b = _b.grad.numpy()
        result_grad_a = np.round(result_grad_a, decimals = 5)
        result_grad_b = np.round(result_grad_b, decimals = 5)
        correct_grad_a = np.array([0.], dtype=np.float32)
        correct_grad_b = np.array([0.], dtype=np.float32)
        correct_grad_a = np.round(correct_grad_a, decimals = 5)
        correct_grad_b = np.round(correct_grad_b, decimals = 5)
        assert result_grad_a == correct_grad_a
        assert result_grad_b == correct_grad_b

        print("Test state vector propagation mode backward calculation using parameter shift method ok!")