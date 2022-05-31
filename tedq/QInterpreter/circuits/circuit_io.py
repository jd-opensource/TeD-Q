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
This module contains I/O functions, like we can construct tedq circuit from qasm file.
"""

from tedq.QInterpreter.operators.qubit import *  # pylint: disable=wildcard-import, unused-wildcard-import


_get_gate_class = {
    "x":(PauliX, 1, 0),
    "y":(PauliY, 1, 0),
    "z":(PauliZ, 1, 0),
    "h":(Hadamard, 1, 0),
    "s":(S, 1, 0),
    #"t":(T, 1, 0),
    "x_1_2":(SX, 1, 0),
    "cnot":(CNOT, 2, 0),
    "rx":(RX, 1, 1),
    "ry":(RY, 1, 1),
    "rz":(RZ, 1, 1),
}   

def _convert_ints_and_floats(str_x):
    if isinstance(str_x, str):
        try:
            return int(str_x)
        except ValueError:
            pass    

        try:
            return float(str_x)
        except ValueError:
            pass    

    return str_x    


def parse_qasm(qasm):
    """Parse qasm from a string.
    Parameters
    ----------
    qasm : str
        The full string of the qasm file.
    Returns
    -------
    circuit_info : dict
        Information about the circuit:
        - circuit_info['n']: the number of qubits
        - circuit_info['n_gates']: the number of gates in total
        - circuit_info['gates']: list[list[str]], list of gates, each of which
          is a list of strings read from a line of the qasm file.
    """ 

    #lines = qasm.split('\n')
    lines = [line.rstrip('\n') for line in qasm]
    n_qubits = int(lines[0])   

    # turn into tuples of python types
    gates = [
        tuple(map(_convert_ints_and_floats, line.strip().split(" ")))
        for line in lines[1:] if line
    ]   

    return {
        'n_qubits': n_qubits,
        'gates': gates,
    }   


def parse_qasm_file(fname, **kwargs):
    r"""
    Parse a qasm file.
    """
    encoding=kwargs.pop('encoding', 'utf-8')
    with open(fname, "r", encoding=encoding) as file_handle:
        info = parse_qasm(file_handle, **kwargs)
    #print(info)
    return info 

def convert_to_tedq_format(gate):
    r'''
    Prepare to put a gate in tedq format
    '''
    name = gate[1]
    try:
        (name_class, wires, n_pars) = _get_gate_class[name]
    except KeyError as error:
        raise ValueError("This gate from qasm file is not supported yet!!!") from error
    pos = 2+wires
    qubits = gate[2:pos]
    pars = gate[pos:pos+n_pars]
    return (name_class, qubits, pars)

class FromQasmFile():
    r'''
    A helping class for reading quantum circuit from qasm file to tedq circuit.
    '''
    def __init__(self, fname):
        r"""
        Generate a ``Circuit`` instance from a qasm file.
        """
        info = parse_qasm_file(fname)
        self._n_qubits = info['n_qubits']
        gates = info['gates']   

        self._operands = []
        for gate in gates:
            operand = convert_to_tedq_format(gate)
            self._operands.append(operand)    
  

    def __call__(self):
        r'''
        Call this function, then the operators will be stored on "_active_warehouse".
        And then those operators will be used to construct tedq quantum circuit.
        '''
        for operand in self._operands:
            (name_class, qubits, pars) = operand
            name_class(*pars, qubits=qubits)

    @property
    def n_qubits(self):
        r'''
        return number of qubits of qasm quantum circuit
        '''
        return self._n_qubits
    
