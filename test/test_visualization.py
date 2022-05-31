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
import tedq as qai
import random


class TestVisulization:
    r"""Test visualization with matplotlib library
    """
    def test_initialization(self):
        params = tuple([random.random() for _ in range(2)])
        def circuitDef(*params):
            qai.RY(params[0], qubits=[0])
            qai.RZ(params[1], qubits=[1])
            return qai.expval(qai.PauliZ(qubits=[0]))
        circuit = qai.Circuit(circuitDef, 2, *params)

        # Test basic property

        drawer = qai.matplotlib_drawer(circuit)
        assert drawer._plt.__name__ == "matplotlib.pyplot"
        assert drawer._circuit is circuit
        assert drawer._qubit_num is circuit._num_qubits

        # Test operators in the circuit

        layers = drawer.get_layers(circuit)
        countGate = 0
        for layer in layers:
            countGate+=len(layer)
        assert countGate == len(circuit.operators)

        # Test measurements
        
        countGate = 0
        for layer in layers:
            #print(layer)
            countGate+=1
        assert countGate == len(circuit.measurements)

        print("Test matplotlib_drawer function ok!")
