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


class TestTemplate:
    r"""Test visualization with matplotlib library
    """
    def test_initialization(self):
        # Global variable
        n_qubits = 6
        depth=1

        ### Fully connected layer
        rand_params = np.random.uniform(high=2 * np.pi, size=((depth+1)*3, n_qubits))

        def circuitDef():
            qai.Templates.FullyConnected(n_qubits, depth, rand_params)
            exp_vals = [qai.measurement.expval(qai.PauliZ(qubits=[position])) for position in range(n_qubits)] 


        circuitFC = qai.Circuit(circuitDef, n_qubits, torch.zeros(n_qubits))

        assert len(circuitFC._operators) is ((depth+1)*3*n_qubits)

        ### Hardware efficienct layer
        rand_params = np.random.uniform(high=2 * np.pi, size=((depth+1)*3, n_qubits))

        def circuitDef():
            qai.Templates.HardwareEfficient(n_qubits, depth, rand_params)
            exp_vals = [qai.measurement.expval(qai.PauliZ(qubits=[position])) for position in range(n_qubits)] 


        circuitHE = qai.Circuit(circuitDef, n_qubits, torch.zeros(n_qubits))

        assert len(circuitHE._operators) is ((depth+1)*3*n_qubits)

        print("Test template ok!")
