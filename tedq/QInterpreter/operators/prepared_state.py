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

# pylint: disable=line-too-long, trailing-whitespace, too-many-lines

r"""
This module contains state initialization classes.
"""

import cmath
import math
import numpy as np
from copy import deepcopy

from .ops_abc import OperatorBase


class IintStateVector(OperatorBase):
    r"""StateVector(qubits)
    StateVector state preparation.
    The matrix form is:


    **Properties**
        * # of qubits: Must equal to number of qubits of the quantum circuit.
        * # of parameters: 0

    Args:
    """
    _num_params = 0
    _num_qubits = 0

    def __init__(self, matrix, qubits: list = [], do_queue: bool = True, **kwargs):


        self._user_defined_matrix = matrix

        super().__init__(qubits=qubits, do_queue=do_queue, is_preparation=True, **kwargs)


    def get_matrix(self):
        r"""Get the matrix representation of an operator instance in computational basis
        """
        return self._user_defined_matrix


