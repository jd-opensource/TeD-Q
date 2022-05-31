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

import tedq as qai
from tedq.optimizers.gradient_descent import GradientDescentOptimizer

class Test_gradient_descent():
    r"""
    """
    def test_construction(self):
        #
        def circuitDef(*params):
            qai.RX(params[0], qubits=[0])
            qai.RY(params[1], qubits=[0])
            #return [qai.state()]
            return qai.expval(qai.PauliZ(qubits=[0]))

        circuit = qai.Circuit(circuitDef, 6, 0.54, 0.12)
        my_compilecircuit = circuit.compilecircuit(backend="jax")

        def cost(*params):
            return my_compilecircuit(*params)[0]

        Optimizer = qai.GradientDescentOptimizer(cost, [0, 1], 0.4, interface="jax")

        assert isinstance(Optimizer, GradientDescentOptimizer)
        assert Optimizer._interface == "jax"
        assert Optimizer._trainable_params == [0,1]
        assert Optimizer._stepsize == 0.4
        assert Optimizer.objective_fn == cost

        print("test gradient descent optimizer construction ok!")

    def test_change_step_size(self):
        #
        def circuitDef(*params):
            qai.RX(params[0], qubits=[0])
            qai.RY(params[1], qubits=[0])
            #return [qai.state()]
            return qai.expval(qai.PauliZ(qubits=[0]))

        circuit = qai.Circuit(circuitDef, 6, 0.54, 0.12)
        my_compilecircuit = circuit.compilecircuit(backend="jax")

        def cost(*params):
            return my_compilecircuit(*params)[0]

        Optimizer = qai.GradientDescentOptimizer(cost, [0, 1], 0.4, interface="jax")

        assert Optimizer._stepsize == 0.4
        Optimizer.update_stepsize(0.21)
        assert Optimizer._stepsize == 0.21

        print("test gradient descent optimizer update step size ok!")

    def test_with_qubit_rotation_example(self):
        """
        """
        # define the quantum circuit
        def circuitDef(*params):
            qai.RX(params[0], qubits=[0])
            qai.RY(params[1], qubits=[0])
            #return [qai.state()]
            return qai.expval(qai.PauliZ(qubits=[0]))
        # transfer to quantum circuit
        circuit = qai.Circuit(circuitDef, 6, 0.54, 0.12)
        # compiled the circuit using pytorch backend and built-in back propagation method
        # and using default state vector propagation mode
        my_compilecircuit = circuit.compilecircuit(backend="pytorch")
        # define cost function
        def cost(*params):
            return my_compilecircuit(*params)[0]

        # initial the optimizer
        Optimizer = qai.GradientDescentOptimizer(cost, [0, 1], 0.4, interface="pytorch")

        import torch
        #parameters to be trained
        a = torch.tensor([0.011], dtype = torch.float32, requires_grad = True)
        b = torch.tensor([0.012], dtype = torch.float32, requires_grad = True)
        my_params = (a, b)
        new_params = my_params
        #train the parameters
        for i in range(100):
            new_params = Optimizer.step(*new_params)
        # get the final result using trained parameter
        result = cost(*new_params)

        # obtain trained parameters and final result in numpy format
        param_1 = new_params[0].detach().numpy()
        param_2 = new_params[1].detach().numpy()
        result = result.detach().numpy()
        # up to 5 decimals accuracy
        param_1 = np.round(param_1, decimals=5)
        param_2 = np.round(param_2, decimals=5)
        result = np.round(result, decimals=5)
        # correct result
        correct_param1 = np.array([7.152645e-18], dtype=np.float32)
        correct_param2 = np.array([3.1415925], dtype=np.float32)
        correct_result = np.array([-1.], dtype=np.float32)
        # up to 5 decimals accuracy
        correct_param1 = np.round(correct_param1, decimals=5)
        correct_param2 = np.round(correct_param2, decimals=5)
        correct_result = np.round(correct_result, decimals=5)
        # assertion judgement
        assert param_1 == correct_param1
        assert param_2 == correct_param2
        assert result == correct_result

        print("Test gradient descent optimizer using qubit rotation example ok!")


class Test_pytorch_built_in_optimizer():
    """
    """
    def test_with_qubit_rotation_example(self):
        # define the quantum circuit
        def circuitDef(*params):
            qai.RX(params[0], qubits=[0])
            qai.RY(params[1], qubits=[0])
            #return [qai.state()]
            return qai.expval(qai.PauliZ(qubits=[0]))
        # transfer to quantum circuit
        circuit = qai.Circuit(circuitDef, 6, 0.54, 0.12)
        # compiled the circuit using pytorch backend and built-in back propagation method
        # and using default state vector propagation mode
        my_compilecircuit = circuit.compilecircuit(backend="pytorch")
        # define cost function
        def cost(*params):
            return my_compilecircuit(*params)[0]

        import torch
        #parameters to be trained
        a = torch.tensor([0.011], dtype = torch.float32, requires_grad = True)
        b = torch.tensor([0.012], dtype = torch.float32, requires_grad = True)
        # initial the optimizer
        from torch import optim
        optimizer = optim.Adam([a, b], lr=0.1)
        my_params = (a, b)
        #train the parameters
        for i in range(500):
            optimizer.zero_grad()
            #print(b.grad)
            loss = cost(*my_params)
            loss.backward()
            optimizer.step()
            #print(my_params)
        # get the final result using trained parameter
        result = cost(*my_params)

        # obtain trained parameters and final result in numpy format
        param_1 = my_params[0].detach().numpy()
        param_2 = my_params[1].detach().numpy()
        result = result.detach().numpy()
        # up to 5 decimals accuracy
        param_1 = np.round(param_1, decimals=5)
        param_2 = np.round(param_2, decimals=5)
        result = np.round(result, decimals=5)
        # correct result
        correct_param1 = np.array([7.152645e-18], dtype=np.float32)
        correct_param2 = np.array([3.1415925], dtype=np.float32)
        correct_result = np.array([-1.], dtype=np.float32)
        # up to 5 decimals accuracy
        correct_param1 = np.round(correct_param1, decimals=5)
        correct_param2 = np.round(correct_param2, decimals=5)
        correct_result = np.round(correct_result, decimals=5)
        # assertion judgement
        assert param_1 == correct_param1
        assert param_2 == correct_param2
        assert result == correct_result

        print("Test pytorch built-in optimizer using qubit rotation example ok!")