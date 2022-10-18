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
This module contains the available quantum gates and observables supported
by QInterpreter, as well as their simple properties.
"""

import cmath
import math
import numpy as np
from copy import deepcopy

from .ops_abc import ObservableBase, GateBase

INV_SQRT2 = 1 / math.sqrt(2)
PI = np.pi

M1 = INV_SQRT2 * (np.sqrt(2) + 1) / 4
M2 = INV_SQRT2 * (np.sqrt(2) - 1) / 4
PHI = 1.
SHIFT1 = np.pi / 2.
SHIFT2 = 3. * np.pi / 2.
CONTROL_GRAD_RECIPE = [
    [M1, PHI, SHIFT1],
    [-1.0*M1, PHI, -SHIFT1],
    [-1.0*M2, PHI, SHIFT2],
    [M2, PHI, -SHIFT2]
]

class Hadamard(ObservableBase, GateBase):
    r"""Hadamard(qubits)
    Hadamard gate operator.
    The matrix form is:

        .. math::
            \begin{align}
                H = \frac{1}{\sqrt{2}}
                \begin{bmatrix}
                    1 & 1 \\
                    1 & -1
                \end{bmatrix}
            \end{align}

    **Properties**
        * # of qubits: 1
        * # of parameters: 0

    Args:
        qubits (List[int]): The qubits this gate acts on
    """
    _num_params = 0
    _num_qubits = 1

    def get_matrix(self):
        r"""Get the matrix representation of an operator instance in computational basis
        """
        return np.array([[INV_SQRT2, INV_SQRT2], [INV_SQRT2, -INV_SQRT2]])

    def diagonalizing_gates(self, do_queue=False):
        r"""Return list of unitary operators that can diagonalize this gate.
        It can be used to rotate the specified qubits so that let the state
        in the eigenbasis of the Hadamard operator.

        For the Hadamard operator,

        .. math:: H = U^\dagger H U

        where :math:`U = R_y(-\pi/4)`.

        .. math:: \langle \mathbf{\alpha} \vert H \vert \mathbf{\beta} \rangle =
                  \langle \mathbf{\alpha} \vert U U^\dagger H U U^\dagger \vert \mathbf{\beta} \rangle

        Returns:
            list(~.GateBase): A list of operators that diagonalize Hadamard in
            the computational basis.
        """
        return [RY(-PI / 4, qubits=self.qubits, do_queue=do_queue)]

    @staticmethod
    def decomposition(qubits, do_queue=False):
        """Decompose this operator into products of other operators.

        Args:
            params (tuple[float, int, array]): operator parameters
            qubits (list[int]): qubits the operator acts on
            do_queue (bool): wether put the decomposed operators into queuing.
        
        Returns:
            list[~.GateBase]
        """
        decomp_ops = [
            PhaseShift(PI / 2, qubits=qubits, do_queue=do_queue),
            RX(PI / 2, qubits=qubits, do_queue=do_queue),
            PhaseShift(PI / 2, qubits=qubits, do_queue=do_queue),
        ]
        return decomp_ops

    def adjoint(self, do_queue=False):
        """Get the adjoint operator of this gate operator.

        Adjoint operator tensor is the transposed conjugated tensor of the original gate operator tensor.
        For unitary gate operator, the tensor of adjoint operator is equivalent to the inverse tensor of original gate operator tensor.

        Args:
            do_queue (bool): Optional, default is False. Indicates whether this operator will appear in the quantum circuit.

        Returns:
            GateBase: the adjoint gate

        """
        return Hadamard(qubits=self.qubits, do_queue=do_queue)


class PauliX(ObservableBase, GateBase):
    r"""PauliX(qubits)
    The Pauli X operator.
    The matrix form is:

        .. math::
            \begin{align}
                \sigma_x = 
                \begin{bmatrix}
                    0 & 1 \\
                    1 & 0
                \end{bmatrix}
            \end{align}

    Properties:
        # of qubits: 1
        # of parameters: 0

    Args:
        qubits (List[int] or int): The qubits this gate acts on
    """
    _num_params = 0
    _num_qubits = 1

    def get_matrix(self):
        r"""Get the matrix representation of an operator instance in computational basis
        """
        return np.array([[0, 1], [1, 0]])

    def diagonalizing_gates(self, do_queue=False):
        r"""Return list of unitary operators that can diagonalize this gate.
        It can be used to rotate the specified qubits so that let the state
        in the eigenbasis of the Hadamard operator.

        For the Pauli-X operator,

        .. math:: X = U^\dagger X U

        where :math:`U = H`.

        .. math:: \langle \mathbf{\alpha} \vert X \vert \mathbf{\beta} \rangle = 
                  \langle \mathbf{\alpha} \vert U U^\dagger X U U^\dagger \vert \mathbf{\beta} \rangle

        Returns:
            list(~.GateBase): A list of operators that diagonalize Hadamard in
            the computational basis.
        """
        return [Hadamard(qubits=self.qubits, do_queue=do_queue)]

    @staticmethod
    def decomposition(qubits, do_queue=False):
        """Decompose this operator into products of other operators.

        Args:
            params (tuple[float, int, array]): operator parameters
            qubits (list[int]): qubits the operator acts on
            do_queue (bool): wether put the decomposed operators into queuing.
        
        Returns:
            list[~.GateBase]
        """
        decomp_ops = [
            PhaseShift(PI / 2, qubits=qubits, do_queue=do_queue),
            RX(PI, qubits=qubits, do_queue=do_queue),
            PhaseShift(PI / 2, qubits=qubits, do_queue=do_queue),
        ]
        return decomp_ops

    def adjoint(self, do_queue=False):
        """Get the adjoint operator of this gate operator.

        Adjoint operator tensor is the transposed conjugated tensor of the original gate operator tensor.
        For unitary gate operator, the tensor of adjoint operator is equivalent to the inverse tensor of original gate operator tensor.

        Args:
            do_queue (bool): Optional, default is False. Indicates whether this operator will appear in the quantum circuit.

        Returns:
            GateBase: the adjoint gate

        """
        return PauliX(qubits=self.qubits, do_queue=do_queue)


class PauliY(ObservableBase, GateBase):
    r"""PauliY(qubits)
    The Pauli Y operator.
    The matrix form is:

        .. math::
            \begin{align}
                \sigma_y = 
                \begin{bmatrix}
                    0 & -i \\
                    i & 0
                \end{bmatrix}
            \end{align}

    **Properties:**
        * # of qubits: 1
        * # of parameters: 0

    Args:
        qubits (List[int] or int): The qubits this gate acts on
    """
    _num_params = 0
    _num_qubits = 1

    def get_matrix(self):
        r"""Get the matrix representation of an operator instance in computational basis
        """
        return np.array([[0, -1j], [1j, 0]])

    def diagonalizing_gates(self, do_queue=False):
        r"""Return list of unitary operators that can diagonalize this gate.
        It can be used to rotate the specified qubits so that let the state
        in the eigenbasis of the Hadamard operator.

        For the Pauli-Y operator,

        .. math:: Y = U^\dagger Y U

        where :math:`U = HSZ`.

        .. math:: \langle \mathbf{\alpha} \vert Y \vert \mathbf{\beta} \rangle = 
                  \langle \mathbf{\alpha} \vert U U^\dagger Y U U^\dagger \vert \mathbf{\beta} \rangle

        Returns:
            list(~.GateBase): A list of operators that diagonalize Hadamard in
            the computational basis.
        """
        return [
            PauliZ(qubits=self.qubits, do_queue=do_queue),
            S(qubits=self.qubits, do_queue=do_queue),
            Hadamard(qubits=self.qubits, do_queue=do_queue),
        ]

    @staticmethod
    def decomposition(qubits, do_queue=False):
        """Decompose this operator into products of other operators.
        
        Args:
            params (tuple[float, int, array]): operator parameters
            qubits (list[int]): qubits the operator acts on
            do_queue (bool): wether put the decomposed operators into queuing.
        
        Returns:
            list[~.GateBase]
        """
        decomp_ops = [
            PhaseShift(PI / 2, qubits=qubits, do_queue=do_queue),
            RY(PI, qubits=qubits, do_queue=do_queue),
            PhaseShift(PI / 2, qubits=qubits, do_queue=do_queue),
        ]
        return decomp_ops

    def adjoint(self, do_queue=False):
        """Get the adjoint operator of this gate operator.

        Adjoint operator tensor is the transposed conjugated tensor of the original gate operator tensor.
        For unitary gate operator, the tensor of adjoint operator is equivalent to the inverse tensor of original gate operator tensor.

        Args:
            do_queue (bool): Optional, default is False. Indicates whether this operator will appear in the quantum circuit.

        Returns:
            GateBase: the adjoint gate

        """
        return PauliY(qubits=self.qubits, do_queue=do_queue)


class PauliZ(ObservableBase, GateBase):
    r"""PauliZ(qubits)
    The Pauli Z operator.
    The matrix form is:

        .. math::
            \begin{align}
                \sigma_z = 
                \begin{bmatrix}
                    1 & 0 \\
                    0 & -1
                \end{bmatrix}
            \end{align}

    **Properties:**
        * # of qubits: 1
        * # of parameters: 0

    Args:
        qubits (List[int] or int): The qubits this gate acts on
    """
    _num_params = 0
    _num_qubits = 1

    def get_matrix(self):
        r"""Get the matrix representation of an operator instance in computational basis
        """
        return np.array([[1, 0], [0, -1]])

    def diagonalizing_gates(self, do_queue=False):
        r"""Return list of unitary operators that can diagonalize this gate.
        It can be used to rotate the specified qubits so that let the state
        in the eigenbasis of the Hadamard operator.

        For the Pauli-Z operator,

        .. math:: Z = U^\dagger Z U

        where :math:`U = I`.

        .. math:: \langle \mathbf{\alpha} \vert Z \vert \mathbf{\beta} \rangle = 
                  \langle \mathbf{\alpha} \vert U U^\dagger Z U U^\dagger \vert \mathbf{\beta} \rangle

        Returns:
            list(~.GateBase): A list of operators that diagonalize Hadamard in
            the computational basis.
        """
        return [I(qubits=self.qubits, do_queue=do_queue)]

    @staticmethod
    def decomposition(qubits, do_queue=False):
        """Decompose this operator into products of other operators.

        Args:
            params (tuple[float, int, array]): operator parameters
            qubits (list[int]): qubits the operator acts on
            do_queue (bool): wether put the decomposed operators into queuing.

        Returns:
            list[~.GateBase]
        """
        decomp_ops = [PhaseShift(PI, qubits=qubits, do_queue=do_queue)]
        return decomp_ops

    def adjoint(self, do_queue=False):
        """Get the adjoint operator of this gate operator.

        Adjoint operator tensor is the transposed conjugated tensor of the original gate operator tensor.
        For unitary gate operator, the tensor of adjoint operator is equivalent to the inverse tensor of original gate operator tensor.

        Args:
            do_queue (bool): Optional, default is False. Indicates whether this operator will appear in the quantum circuit.

        Returns:
            GateBase: the adjoint gate

        """
        return PauliZ(qubits=self.qubits, do_queue=do_queue)


class I(ObservableBase, GateBase):  # pylint: disable=invalid-name
    r"""I(qubits)
    The Identiy operator.
    The matrix form is:

        .. math::
            \begin{align}
                I = 
                \begin{bmatrix}
                    1 & 0 \\
                    0 & 1
                \end{bmatrix}
            \end{align}

    **Properties:**
        * # of qubits: 1
        * # of parameters: 0

    Args:
        qubits (List[int] or int): The qubits this gate acts on
    """
    _num_params = 0
    _num_qubits = 1

    def get_matrix(self):
        r"""Get the matrix representation of an operator instance in computational basis
        """
        return np.array([[1, 0], [0, 1]])

    def diagonalizing_gates(self, do_queue=False):
        return [I(qubits=self.qubits, do_queue=do_queue)]

    @staticmethod
    def decomposition(qubits, do_queue=False):
        """Decompose this operator into products of other operators.

        Args:
            params (tuple[float, int, array]): operator parameters
            qubits (list[int]): qubits the operator acts on
            do_queue (bool): wether put the decomposed operators into queuing.
        
        Returns:
            list[~.GateBase]
        """
        decomp_ops = [PhaseShift(PI, qubits=qubits, do_queue=do_queue)]
        return decomp_ops

    def adjoint(self, do_queue=False):
        """Get the adjoint operator of this gate operator.

        Adjoint operator tensor is the transposed conjugated tensor of the original gate operator tensor.
        For unitary gate operator, the tensor of adjoint operator is equivalent to the inverse tensor of original gate operator tensor.

        Args:
            do_queue (bool): Optional, default is False. Indicates whether this operator will appear in the quantum circuit.

        Returns:
            GateBase: the adjoint gate

        """
        return I(qubits=self.qubits, do_queue=do_queue)


class S(GateBase):  # pylint: disable=invalid-name
    r"""S(qubits)
    The single-qubit phase gate.
    The matrix form is:

        .. math::
            \begin{align}
                S = 
                \begin{bmatrix}
                    1 & 0 \\
                    0 & i
                \end{bmatrix}
            \end{align}

    **Properties:**
        * # of qubits: 1
        * # of parameters: 0

    Args:
        qubits (List[int] or int): The qubits this gate acts on
    """
    _num_params = 0
    _num_qubits = 1

    def get_matrix(self):
        r"""Get the matrix representation of an operator instance in computational basis
        """
        return np.array([[1, 0], [0, 1j]])

    @staticmethod
    def decomposition(qubits, do_queue=False):
        """Decompose this operator into products of other operators.

        Args:
            params (tuple[float, int, array]): operator parameters
            qubits (list[int]): qubits the operator acts on
            do_queue (bool): wether put the decomposed operators into queuing.
        
        Returns:
            list[~.GateBase]
        """
        decomp_ops = [PhaseShift(PI / 2, qubits=qubits, do_queue=do_queue)]
        return decomp_ops

    def adjoint(self, do_queue=False):
        """Get the adjoint operator of this gate operator.

        Adjoint operator tensor is the transposed conjugated tensor of the original gate operator tensor.
        For unitary gate operator, the tensor of adjoint operator is equivalent to the inverse tensor of original gate operator tensor.

        Args:
            do_queue (bool): Optional, default is False. Indicates whether this operator will appear in the quantum circuit.

        Returns:
            GateBase: the adjoint gate

        """
        return S(qubits=self.qubits, do_queue=do_queue).inverse()


class T(GateBase):  # pylint: disable=invalid-name
    r"""T(qubits)
    The single-qubit T gate.
    The matrix form is:

        .. math::
            \begin{align}
                T = 
                \begin{bmatrix}
                    1 & 0 \\
                    0 & e^{\frac{i\pi}{4}}
                \end{bmatrix}
            \end{align}

    **Properties:**
        * # of qubits: 1
        * # of parameters: 0

    Args:
        qubits (List[int] or int): The qubits this gate acts on
    """
    _num_params = 0
    _num_qubits = 1

    def get_matrix(self):
        r"""Get the matrix representation of an operator instance in computational basis
        """
        return np.array([[1, 0], [0, cmath.exp(1j * PI / 4)]])

    @staticmethod
    def decomposition(qubits, do_queue=False):
        """Decompose this operator into products of other operators.

        Args:
            params (tuple[float, int, array]): operator parameters
            qubits (list[int]): qubits the operator acts on
            do_queue (bool): wether put the decomposed operators into queuing.
        
        Returns:
            list[~.GateBase]
        """
        decomp_ops = [PhaseShift(PI / 4, qubits=qubits, do_queue=do_queue)]
        return decomp_ops

    def adjoint(self, do_queue=False):
        """Get the adjoint operator of this gate operator.

        Adjoint operator tensor is the transposed conjugated tensor of the original gate operator tensor.
        For unitary gate operator, the tensor of adjoint operator is equivalent to the inverse tensor of original gate operator tensor.

        Args:
            do_queue (bool): Optional, default is False. Indicates whether this operator will appear in the quantum circuit.

        Returns:
            GateBase: the adjoint gate

        """
        return T(qubits=self.qubits, do_queue=do_queue).inverse()


class SX(GateBase):
    r"""SX(qubits)
    The single-qubit Square-Root X gate.
    The matrix form is:

        .. math::
            \begin{align}
                SX = \sqrt{X} = \frac{1}{2}
                \begin{bmatrix}
                    1+i & 1-i \\
                    1-i & 1+i
                \end{bmatrix}
            \end{align}

    **Properties:**
        * # of qubits: 1
        * # of parameters: 0

    Args:
        qubits (List[int] or int): The qubits this gate acts on
    """
    _num_params = 0
    _num_qubits = 1

    def get_matrix(self):
        r"""Get the matrix representation of an operator instance in computational basis
        """
        return 0.5 * np.array([[1 + 1j, 1 - 1j], [1 - 1j, 1 + 1j]])

    @staticmethod
    def decomposition(qubits, do_queue=False):
        """Decompose this operator into products of other operators.

        Args:
            params (tuple[float, int, array]): operator parameters
            qubits (list[int]): qubits the operator acts on
            do_queue (bool): wether put the decomposed operators into queuing.
        
        Returns:
            list[~.GateBase]
        """
        decomp_ops = [
            RZ(PI / 2, qubits=qubits),
            RY(PI / 2, qubits=qubits),
            RZ(-PI, qubits=qubits),
            PhaseShift(PI / 2, qubits=qubits, do_queue=do_queue),
        ]
        return decomp_ops

    def adjoint(self, do_queue=False):
        """Get the adjoint operator of this gate operator.

        Adjoint operator tensor is the transposed conjugated tensor of the original gate operator tensor.
        For unitary gate operator, the tensor of adjoint operator is equivalent to the inverse tensor of original gate operator tensor.

        Args:
            do_queue (bool): Optional, default is False. Indicates whether this operator will appear in the quantum circuit.

        Returns:
            GateBase: the adjoint gate

        """
        return SX(qubits=self.qubits, do_queue=do_queue).inverse()


class CNOT(GateBase):
    r"""CNOT(qubits)
    The controlled-NOT (CX) gate.
    The matrix form is:

        .. math::
            \begin{align}
                CNOT = 
                \begin{bmatrix}
                    1 & 0 & 0 & 0 \\
                    0 & 1 & 0 & 0\\
                    0 & 0 & 0 & 1\\
                    0 & 0 & 1 & 0
                \end{bmatrix}
            \end{align}

    .. note:: The first qubit is the **control qubit**.

    **Properties:**
        * # of qubits: 2
        * # of parameters: 0

    Args:
        qubits (List[int]): The qubits the operation acts on
    """
    _num_params = 0
    _num_qubits = 2

    def get_matrix(self):
        r"""Get the matrix representation of an operator instance in computational basis
        """
        return np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]])

    def adjoint(self, do_queue=False):
        """Get the adjoint operator of this gate operator.

        Adjoint operator tensor is the transposed conjugated tensor of the original gate operator tensor.
        For unitary gate operator, the tensor of adjoint operator is equivalent to the inverse tensor of original gate operator tensor.

        Args:
            do_queue (bool): Optional, default is False. Indicates whether this operator will appear in the quantum circuit.

        Returns:
            GateBase: the adjoint gate

        """
        return CNOT(qubits=self.qubits, do_queue=do_queue)


class CZ(GateBase):
    r"""CZ(qubits)
    The controlled-Z gate.
    The matrix form is:

        .. math::
            \begin{align}
                CZ = 
                \begin{bmatrix}
                    1 & 0 & 0 & 0 \\
                    0 & 1 & 0 & 0\\
                    0 & 0 & 1 & 0\\
                    0 & 0 & 0 & -1
                \end{bmatrix}
            \end{align}

    .. note:: The first qubit is the **control qubit**.

    **Properties:**
        * # of qubits: 2
        * # of parameters: 0

    Args:
        qubits (List[int]): The qubits the operation acts on
    """
    _num_params = 0
    _num_qubits = 2

    def get_matrix(self):
        r"""Get the matrix representation of an operator instance in computational basis
        """
        return np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, -1]])

    def adjoint(self, do_queue=False):
        """Get the adjoint operator of this gate operator.

        Adjoint operator tensor is the transposed conjugated tensor of the original gate operator tensor.
        For unitary gate operator, the tensor of adjoint operator is equivalent to the inverse tensor of original gate operator tensor.

        Args:
            do_queue (bool): Optional, default is False. Indicates whether this operator will appear in the quantum circuit.

        Returns:
            GateBase: the adjoint gate

        """
        return CZ(qubits=self.qubits, do_queue=do_queue)


class CY(GateBase):
    r"""CY(qubits)
    The controlled-Y gate.
    The matrix form is:

        .. math::
            \begin{align}
                CY = 
                \begin{bmatrix}
                    1 & 0 & 0 & 0 \\
                    0 & 1 & 0 & 0\\
                    0 & 0 & 0 & -i\\
                    0 & 0 & i & 0
                \end{bmatrix}
            \end{align}

    .. note:: The first qubit is the **control qubit**.

    **Properties:**
        * # of qubits: 2
        * # of parameters: 0

    Args:
        qubits (List[int]): The qubits the operation acts on
    """
    _num_params = 0
    _num_qubits = 2

    def get_matrix(self):
        r"""Get the matrix representation of an operator instance in computational basis
        """
        return np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, -1j], [0, 0, 1j, 0]])

    @staticmethod
    def decomposition(qubits, do_queue=False):
        """Decompose this operator into products of other operators.

        Args:
            params (tuple[float, int, array]): operator parameters
            qubits (list[int]): qubits the operator acts on
            do_queue (bool): wether put the decomposed operators into queuing.
        
        Returns:
            list[~.GateBase]
        """
        decomp_ops = [CRY(PI, qubits=qubits, do_queue=do_queue), S(qubits=qubits[0], do_queue=do_queue)]
        return decomp_ops

    def adjoint(self, do_queue=False):
        """Get the adjoint operator of this gate operator.

        Adjoint operator tensor is the transposed conjugated tensor of the original gate operator tensor.
        For unitary gate operator, the tensor of adjoint operator is equivalent to the inverse tensor of original gate operator tensor.

        Args:
            do_queue (bool): Optional, default is False. Indicates whether this operator will appear in the quantum circuit.

        Returns:
            GateBase: the adjoint gate

        """
        return CY(qubits=self.qubits, do_queue=do_queue)


class SWAP(GateBase):
    r"""SWAP(qubits)
    The swap gate.
    The matrix form is:

        .. math::
            \begin{align}
                SWAP = 
                \begin{bmatrix}
                    1 & 0 & 0 & 0 \\
                    0 & 0 & 1 & 0\\
                    0 & 1 & 0 & 0\\
                    0 & 0 & 0 & 1
                \end{bmatrix}
            \end{align}


    **Properties:**
        * # of qubits: 2
        * # of parameters: 0

    Args:
        qubits (List[int]): The qubits the operation acts on
    """
    _num_params = 0
    _num_qubits = 2

    def get_matrix(self):
        r"""Get the matrix representation of an operator instance in computational basis
        """
        return np.array([[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]])

    def adjoint(self, do_queue=False):
        """Get the adjoint operator of this gate operator.

        Adjoint operator tensor is the transposed conjugated tensor of the original gate operator tensor.
        For unitary gate operator, the tensor of adjoint operator is equivalent to the inverse tensor of original gate operator tensor.

        Args:
            do_queue (bool): Optional, default is False. Indicates whether this operator will appear in the quantum circuit.

        Returns:
            GateBase: the adjoint gate

        """
        return SWAP(qubits=self.qubits, do_queue=do_queue)


class CSWAP(GateBase):
    r"""CSWAP(qubits)
    The controlled-swap gate.
    The matrix form is:

        .. math::
            \begin{align}
                CSWAP = 
                \begin{bmatrix}
                    1 & 0 & 0 & 0 & 0 & 0 & 0 & 0 \\
                    0 & 1 & 0 & 0 & 0 & 0 & 0 & 0 \\
                    0 & 0 & 1 & 0 & 0 & 0 & 0 & 0 \\
                    0 & 0 & 0 & 1 & 0 & 0 & 0 & 0 \\
                    0 & 0 & 0 & 0 & 1 & 0 & 0 & 0 \\
                    0 & 0 & 0 & 0 & 0 & 0 & 1 & 0 \\
                    0 & 0 & 0 & 0 & 0 & 1 & 0 & 0 \\
                    0 & 0 & 0 & 0 & 0 & 0 & 0 & 1
                \end{bmatrix}
            \end{align}

    .. note:: The first qubit is the **control qubit**.

    **Properties:**
        * # of qubits: 3
        * # of parameters: 0

    Args:
        qubits (List[int]): The qubits the operation acts on
    """
    _num_params = 0
    _num_qubits = 3

    def get_matrix(self):
        r"""Get the matrix representation of an operator instance in computational basis
        """
        return np.array(
            [
                [1, 0, 0, 0, 0, 0, 0, 0],
                [0, 1, 0, 0, 0, 0, 0, 0],
                [0, 0, 1, 0, 0, 0, 0, 0],
                [0, 0, 0, 1, 0, 0, 0, 0],
                [0, 0, 0, 0, 1, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 1, 0],
                [0, 0, 0, 0, 0, 1, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 1],
            ]
        )

    def adjoint(self, do_queue=False):
        """Get the adjoint operator of this gate operator.

        Adjoint operator tensor is the transposed conjugated tensor of the original gate operator tensor.
        For unitary gate operator, the tensor of adjoint operator is equivalent to the inverse tensor of original gate operator tensor.

        Args:
            do_queue (bool): Optional, default is False. Indicates whether this operator will appear in the quantum circuit.

        Returns:
            GateBase: the adjoint gate

        """
        return CSWAP(qubits=self.qubits, do_queue=do_queue)


class Toffoli(GateBase):
    r"""Toffoli(qubits)
    Toffoli (controlled-controlled-X) gate.
    The matrix form is:

        .. math::
            \begin{align}
                Toffoli = 
                \begin{bmatrix}
                    1 & 0 & 0 & 0 & 0 & 0 & 0 & 0\\
                    0 & 1 & 0 & 0 & 0 & 0 & 0 & 0\\
                    0 & 0 & 1 & 0 & 0 & 0 & 0 & 0\\
                    0 & 0 & 0 & 1 & 0 & 0 & 0 & 0\\
                    0 & 0 & 0 & 0 & 1 & 0 & 0 & 0\\
                    0 & 0 & 0 & 0 & 0 & 1 & 0 & 0\\
                    0 & 0 & 0 & 0 & 0 & 0 & 0 & 1\\
                    0 & 0 & 0 & 0 & 0 & 0 & 1 & 0
                \end{bmatrix}
            \end{align}

    **Properties:**
        * # of qubits: 3
        * # of parameters: 0

    Args:
        qubits (List[int]): the subsystem the gate acts on
    """
    _num_params = 0
    _num_qubits = 3

    def get_matrix(self):
        r"""Get the matrix representation of an operator instance in computational basis
        """
        return np.array(
            [
                [1, 0, 0, 0, 0, 0, 0, 0],
                [0, 1, 0, 0, 0, 0, 0, 0],
                [0, 0, 1, 0, 0, 0, 0, 0],
                [0, 0, 0, 1, 0, 0, 0, 0],
                [0, 0, 0, 0, 1, 0, 0, 0],
                [0, 0, 0, 0, 0, 1, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 1],
                [0, 0, 0, 0, 0, 0, 1, 0],
            ]
        )

    def adjoint(self, do_queue=False):
        """Get the adjoint operator of this gate operator.

        Adjoint operator tensor is the transposed conjugated tensor of the original gate operator tensor.
        For unitary gate operator, the tensor of adjoint operator is equivalent to the inverse tensor of original gate operator tensor.

        Args:
            do_queue (bool): Optional, default is False. Indicates whether this operator will appear in the quantum circuit.

        Returns:
            GateBase: the adjoint gate

        """
        return Toffoli(qubits=self.qubits, do_queue=do_queue)


class RX(GateBase):
    r"""
    The single qubit X rotation gate.
    The matrix form is:

        .. math::
            \begin{align}
                R_x(\phi) = e^{-i\phi\sigma_x/2} =
                \begin{bmatrix}
                    \cos(\phi/2) & -i\sin(\phi/2) \\
                    -i\sin(\phi/2) & \cos(\phi/2)
                \end{bmatrix}
            \end{align}

    **Properties:**
        * # of qubits: 1
        * # of parameters: 1

    Args:
        phi (float): rotation angle :math:`\phi`
        qubits (List[int] or int): The qubits this gate acts on
    """
    _num_params = 1
    _num_qubits = 1
    #grad_style = "A"

    def get_matrix(self):
        r"""Get the matrix representation of an operator instance in computational basis
        """
        theta = self._params[0]
        c = math.cos(theta / 2)  # pylint: disable=invalid-name
        js = 1j * math.sin(-theta / 2)  # pylint: disable=invalid-name

        return np.array([[c, js], [js, c]])

    def adjoint(self, do_queue=False):
        """Get the adjoint operator of this gate operator.

        Adjoint operator tensor is the transposed conjugated tensor of the original gate operator tensor.
        For unitary gate operator, the tensor of adjoint operator is equivalent to the inverse tensor of original gate operator tensor.

        Args:
            do_queue (bool): Optional, default is False. Indicates whether this operator will appear in the quantum circuit.

        Returns:
            GateBase: the adjoint gate

        """
        return RX(-self.parameters[0], qubits=self.qubits, do_queue=do_queue)


class RY(GateBase):
    r"""
    The single qubit Y rotation gate.
    The matrix form is:

        .. math::
            \begin{align}
                R_y(\phi) = e^{-i\phi\sigma_y/2} =
                \begin{bmatrix}
                    \cos(\phi/2) & -\sin(\phi/2) \\
                    \sin(\phi/2) & \cos(\phi/2)
                \end{bmatrix}
            \end{align}

    **Properties:**
        * # of qubits: 1
        * # of parameters: 1

    Args:
        phi (float): rotation angle :math:`\phi`
        qubits (List[int] or int): The qubits this gate acts on
    """
    _num_params = 1
    _num_qubits = 1
    #grad_style = "A"

    def get_matrix(self):
        r"""Get the matrix representation of an operator instance in computational basis
        """
        theta = self._params[0]
        c = math.cos(theta / 2)  # pylint: disable=invalid-name
        s = math.sin(theta / 2)  # pylint: disable=invalid-name

        return np.array([[c, -s], [s, c]])

    def adjoint(self, do_queue=False):
        """Get the adjoint operator of this gate operator.

        Adjoint operator tensor is the transposed conjugated tensor of the original gate operator tensor.
        For unitary gate operator, the tensor of adjoint operator is equivalent to the inverse tensor of original gate operator tensor.

        Args:
            do_queue (bool): Optional, default is False. Indicates whether this operator will appear in the quantum circuit.

        Returns:
            GateBase: the adjoint gate

        """
        return RY(-self.parameters[0], qubits=self.qubits, do_queue=do_queue)


class RZ(GateBase):
    r"""
    The single qubit Z rotation gate.
    The matrix form is:

        .. math::
            \begin{align}
                R_z(\phi) = e^{-i\phi\sigma_z/2} =
                \begin{bmatrix}
                    e^{-i\phi/2} & 0 \\
                    0 & e^{i\phi/2}
                \end{bmatrix}
            \end{align}

    **Properties:**
        * # of qubits: 1
        * # of parameters: 1

    Args:
        phi (float): rotation angle :math:`\phi`
        qubits (List[int] or int): The qubits this gate acts on
    """
    _num_params = 1
    _num_qubits = 1
    #grad_style = "A"

    def get_matrix(self):
        r"""Get the matrix representation of an operator instance in computational basis
        """
        theta = self._params[0]
        p = cmath.exp(-0.5j * theta)  # pylint: disable=invalid-name

        return np.array([[p, 0], [0, p.conjugate()]])

    def adjoint(self, do_queue=False):
        """Get the adjoint operator of this gate operator.

        Adjoint operator tensor is the transposed conjugated tensor of the original gate operator tensor.
        For unitary gate operator, the tensor of adjoint operator is equivalent to the inverse tensor of original gate operator tensor.

        Args:
            do_queue (bool): Optional, default is False. Indicates whether this operator will appear in the quantum circuit.

        Returns:
            GateBase: the adjoint gate

        """
        return RZ(-self.parameters[0], qubits=self.qubits, do_queue=do_queue)


class Rot(GateBase):
    r"""
    Arbitrary single qubit rotation gate.
    The matrix form is:

        .. math::
            \begin{align}
                R(\phi,\theta,\omega) = RZ(\omega)RY(\theta)RZ(\phi)= 
                \begin{bmatrix}
                    e^{-i(\phi+\omega)/2}\cos(\theta/2) & -e^{i(\phi-\omega)/2}\sin(\theta/2) \\
                    e^{-i(\phi-\omega)/2}\sin(\theta/2) & e^{i(\phi+\omega)/2}\cos(\theta/2)
                \end{bmatrix}
            \end{align}

    **Properties:**
        * # of qubits: 1
        * # of parameters: 3

    .. note::

        This ``Rot`` gate can be decomposed into :class:`~.RZ` and :class:`~.RY` gates.

    Args:
        phi (float): rotation angle :math:`\phi`
        theta (float): rotation angle :math:`\theta`
        omega (float): rotation angle :math:`\omega`
        qubits (List[int] or int): The qubits this gate acts on
    """
    _num_params = 3
    _num_qubits = 1
    #grad_style = "A"

    def get_matrix(self):
        r"""Get the matrix representation of an operator instance in computational basis
        """
        phi = self._params[0]
        theta = self._params[1]
        omega = self._params[2]
        c = math.cos(theta / 2)  # pylint: disable=invalid-name
        s = math.sin(theta / 2)  # pylint: disable=invalid-name

        return np.array(
            [
                [
                    cmath.exp(-0.5j * (phi + omega)) * c,
                    -cmath.exp(0.5j * (phi - omega)) * s,
                ],
                [
                    cmath.exp(-0.5j * (phi - omega)) * s,
                    cmath.exp(0.5j * (phi + omega)) * c,
                ],
            ]
        )

    @staticmethod
    def decomposition(phi, theta, omega, qubits, do_queue=False):
        """Decompose this operator into products of other operators.

        Args:
            params (tuple[float, int, array]): operator parameters
            qubits (list[int]): qubits the operator acts on
            do_queue (bool): wether put the decomposed operators into queuing.
        
        Returns:
            list[~.GateBase]
        """
        decomp_ops = [
            RZ(phi, qubits=qubits, do_queue=do_queue),
            RY(theta, qubits=qubits, do_queue=do_queue),
            RZ(omega, qubits=qubits, do_queue=do_queue),
        ]
        return decomp_ops

    def adjoint(self, do_queue=False):
        """Get the adjoint operator of this gate operator.

        Adjoint operator tensor is the transposed conjugated tensor of the original gate operator tensor.
        For unitary gate operator, the tensor of adjoint operator is equivalent to the inverse tensor of original gate operator tensor.

        Args:
            do_queue (bool): Optional, default is False. Indicates whether this operator will appear in the quantum circuit.

        Returns:
            GateBase: the adjoint gate

        """
        phi, theta, omega = self.parameters
        return Rot(-omega, -theta, -phi, qubits=self.qubits, do_queue=do_queue)


class PhaseShift(GateBase):
    r"""PhaseShift(phi, wires)
    Arbitrary single qubit local phase shift gate.
    The matrix form is:

        .. math::
            \begin{align}
                R_\phi(\phi) = e^{i\phi/2}R_z(\phi) =
                \begin{bmatrix}
                    1 & 0 \\
                    0 & e^{i\phi}
                \end{bmatrix}
            \end{align}

    **Properties:**
        * # of qubits: 1
        * # of parameters: 1

    Args:
        phi (float): rotation angle :math:`\phi`
        qubits (List[int] or int): The qubits this gate acts on
    """
    _num_params = 1
    _num_qubits = 1
    #grad_style = "A"

    def get_matrix(self):
        r"""Get the matrix representation of an operator instance in computational basis
        """
        phi = self._params[0]
        return np.array([[1, 0], [0, cmath.exp(1j * phi)]])

    @staticmethod
    def decomposition(phi, qubits, do_queue=False):
        """Decompose this operator into products of other operators.

        Args:
            params (tuple[float, int, array]): operator parameters
            qubits (list[int]): qubits the operator acts on
            do_queue (bool): wether put the decomposed operators into queuing.
        
        Returns:
            list[~.GateBase]
        """
        decomp_ops = [RZ(phi, qubits=qubits, do_queue=do_queue)]
        return decomp_ops

    def adjoint(self, do_queue=False):
        """Get the adjoint operator of this gate operator.

        Adjoint operator tensor is the transposed conjugated tensor of the original gate operator tensor.
        For unitary gate operator, the tensor of adjoint operator is equivalent to the inverse tensor of original gate operator tensor.

        Args:
            do_queue (bool): Optional, default is False. Indicates whether this operator will appear in the quantum circuit.

        Returns:
            GateBase: the adjoint gate

        """
        return PhaseShift(-self.parameters[0], qubits=self.qubits, do_queue=do_queue)


class ControlledPhaseShift(GateBase):
    r"""ControlledPhaseShift gate.
    The matrix form is:

        .. math::
            \begin{align}
                CR_\phi(\phi) =
                \begin{bmatrix}
                    1 & 0 & 0 & 0 \\
                    0 & 1 & 0 & 0 \\
                    0 & 0 & 1 & 0 \\
                    0 & 0 & 0 & e^{i\phi}
                \end{bmatrix}
            \end{align}

    .. note:: The first qubit is the **control qubit**.

    **Properties:**
        * # of qubits: 2
        * # of parameters: 1

    Args:
        phi (float): rotation angle :math:`\phi`
        qubits (List[int]): The qubits this gate acts on
    """
    _num_params = 1
    _num_qubits = 2
    #grad_style = "A"

    def get_matrix(self):
        r"""Get the matrix representation of an operator instance in computational basis
        """
        phi = self._params[0]
        return np.array(
            [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0],
                [0, 0, 0, cmath.exp(1j * phi)]]
        )

    @staticmethod
    def decomposition(phi, qubits, do_queue=False):
        """Decompose this operator into products of other operators.

        Args:
            params (tuple[float, int, array]): operator parameters
            qubits (list[int]): qubits the operator acts on
            do_queue (bool): wether put the decomposed operators into queuing.
        
        Returns:
            list[~.GateBase]
        """
        decomp_ops = [
            PhaseShift(phi / 2, qubits=qubits[0], do_queue=do_queue),
            CNOT(qubits=[0, 1], do_queue=do_queue),
            PhaseShift(-phi / 2, qubits=qubits[1], do_queue=do_queue),
            CNOT(qubits=[0, 1], do_queue=do_queue),
            PhaseShift(phi / 2, qubits=qubits[1], do_queue=do_queue),
        ]
        return decomp_ops

    def adjoint(self, do_queue=False):
        """Get the adjoint operator of this gate operator.

        Adjoint operator tensor is the transposed conjugated tensor of the original gate operator tensor.
        For unitary gate operator, the tensor of adjoint operator is equivalent to the inverse tensor of original gate operator tensor.

        Args:
            do_queue (bool): Optional, default is False. Indicates whether this operator will appear in the quantum circuit.

        Returns:
            GateBase: the adjoint gate

        """
        return ControlledPhaseShift(-self.parameters[0], qubits=self.qubits, do_queue=do_queue)


class CRX(GateBase):
    r"""controlled-RX gate.
    The matrix form is:

        .. math::
            \begin{align}
                CR_x(\phi) =
                \begin{bmatrix}
                    & 1 & 0 & 0 & 0 \\
                    & 0 & 1 & 0 & 0\\
                    & 0 & 0 & \cos(\phi/2) & -i\sin(\phi/2)\\
                    & 0 & 0 & -i\sin(\phi/2) & \cos(\phi/2)
                \end{bmatrix}
            \end{align}

    **Properties:**
        * # of qubits: 2
        * # of parameters: 1

    Args:
        phi (float): rotation angle :math:`\phi`
        qubits (List[int]): The qubits this gate acts on
    """
    _num_params = 1
    _num_qubits = 2
    #grad_style = "A"

    def get_parameter_shift(self):
        r"""
        Multiplier and shift for the given parameter.
        By default, differential can be obtain as: :math:`\frac{d}{d\phi}f(\phi) = m1 \left[f(\phi+SHIFT1) - f(\phi-SHIFT1)\right] - m2 \left[f(\phi+SHIFT2) - f(\phi-SHIFT2)\right]`

        Returns: 
            list[list[float]]
        """

        if self.grad_style == "param-shift":  # pylint: disable=no-else-return
            return CONTROL_GRAD_RECIPE
        else:
            raise ValueError(f'{self.name} : This quantum gate do not support parameter-shift method')

    def get_matrix(self):
        r"""Get the matrix representation of an operator instance in computational basis
        """
        theta = self._params[0]
        c = math.cos(theta / 2)  # pylint: disable=invalid-name
        js = 1j * math.sin(-theta / 2)  # pylint: disable=invalid-name

        return np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, c, js], [0, 0, js, c]])

    @staticmethod
    def decomposition(theta, qubits, do_queue=False):
        """Decompose this operator into products of other operators.

        Args:
            params (tuple[float, int, array]): operator parameters
            qubits (list[int]): qubits the operator acts on
            do_queue (bool): wether put the decomposed operators into queuing.

        Returns:
            list[~.GateBase]
        """
        decomp_ops = [
            RZ(PI / 2, qubits=qubits[1], do_queue=do_queue),
            RY(theta / 2, qubits=qubits[1], do_queue=do_queue),
            CNOT(qubits=qubits, do_queue=do_queue),
            RY(-theta / 2, qubits=qubits[1], do_queue=do_queue),
            CNOT(qubits=qubits, do_queue=do_queue),
            RZ(-PI / 2, qubits=qubits[1], do_queue=do_queue),
        ]
        return decomp_ops

    def adjoint(self, do_queue=False):
        """Get the adjoint operator of this gate operator.

        Adjoint operator tensor is the transposed conjugated tensor of the original gate operator tensor.
        For unitary gate operator, the tensor of adjoint operator is equivalent to the inverse tensor of original gate operator tensor.

        Args:
            do_queue (bool): Optional, default is False. Indicates whether this operator will appear in the quantum circuit.

        Returns:
            GateBase: the adjoint gate

        """
        return CRX(-self.parameters[0], qubits=self.qubits, do_queue=do_queue)


class CRY(GateBase):
    r"""controlled-RY gate.
    The matrix form is:

        .. math::
            \begin{align}
                CR_y(\phi) =
                \begin{bmatrix}
                    1 & 0 & 0 & 0 \\
                    0 & 1 & 0 & 0\\
                    0 & 0 & \cos(\phi/2) & -\sin(\phi/2)\\
                    0 & 0 & \sin(\phi/2) & \cos(\phi/2)
                \end{bmatrix}
            \end{align}

    **Properties:**
        * # of qubits: 2
        * # of parameters: 1

    Args:
        phi (float): rotation angle :math:`\phi`
        qubits (List[int]): The qubits this gate acts on
    """
    _num_params = 1
    _num_qubits = 2
    #grad_style = "A"

    def get_parameter_shift(self):
        r"""
        Multiplier and shift for the given parameter.
        By default, differential can be obtain as: :math:`\frac{d}{d\phi}f(\phi) = m1 \left[f(\phi+SHIFT1) - f(\phi-SHIFT1)\right] - m2 \left[f(\phi+SHIFT2) - f(\phi-SHIFT2)\right]`

        Returns: 
            list[list[float]]
        """

        if self.grad_style == "param-shift":  # pylint: disable=no-else-return
            return CONTROL_GRAD_RECIPE
        else:
            raise ValueError(f'{self.name} : This quantum gate do not support parameter-shift method')

    def get_matrix(self):
        r"""Get the matrix representation of an operator instance in computational basis
        """
        theta = self._params[0]
        c = math.cos(theta / 2)  # pylint: disable=invalid-name
        s = math.sin(theta / 2)  # pylint: disable=invalid-name

        return np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, c, -s], [0, 0, s, c]])

    @staticmethod
    def decomposition(theta, qubits, do_queue=False):
        """Decompose this operator into products of other operators.

        Args:
            params (tuple[float, int, array]): operator parameters
            qubits (list[int]): qubits the operator acts on
            do_queue (bool): wether put the decomposed operators into queuing.

        Returns:
            list[~.GateBase]
        """
        decomp_ops = [
            RY(theta / 2, qubits=qubits[1], do_queue=do_queue),
            CNOT(qubits=qubits, do_queue=do_queue),
            RY(-theta / 2, qubits=qubits[1], do_queue=do_queue),
            CNOT(qubits=qubits, do_queue=do_queue),
        ]
        return decomp_ops

    def adjoint(self, do_queue=False):
        """Get the adjoint operator of this gate operator.

        Adjoint operator tensor is the transposed conjugated tensor of the original gate operator tensor.
        For unitary gate operator, the tensor of adjoint operator is equivalent to the inverse tensor of original gate operator tensor.

        Args:
            do_queue (bool): Optional, default is False. Indicates whether this operator will appear in the quantum circuit.

        Returns:
            GateBase: the adjoint gate

        """
        return CRY(-self.parameters[0], qubits=self.qubits, do_queue=do_queue)


class CRZ(GateBase):
    r"""controlled-RZ gate.
    The matrix form is:

        .. math::
            \begin{align}
                CR_z(\phi) =
                \begin{bmatrix}
                    1 & 0 & 0 & 0 \\
                    0 & 1 & 0 & 0\\
                    0 & 0 & e^{-i\phi/2} & 0\\
                    0 & 0 & 0 & e^{i\phi/2}
                \end{bmatrix}
            \end{align}

    **Properties:**
        * # of qubits: 2
        * # of parameters: 1

    Args:
        phi (float): rotation angle :math:`\phi`
        qubits (List[int]): The qubits this gate acts on
    """
    _num_params = 1
    _num_qubits = 2
    #grad_style = "A"

    def get_parameter_shift(self):
        r"""
        Multiplier and shift for the given parameter.
        By default, differential can be obtain as: :math:`\frac{d}{d\phi}f(\phi) = m1 \left[f(\phi+SHIFT1) - f(\phi-SHIFT1)\right] - m2 \left[f(\phi+SHIFT2) - f(\phi-SHIFT2)\right]`

        Returns: 
            list[list[float]]
        """

        if self.grad_style == "param-shift":  # pylint: disable=no-else-return
            return CONTROL_GRAD_RECIPE
        else:
            raise ValueError(f'{self.name} : This quantum gate do not support parameter-shift method')

    def get_matrix(self):
        r"""Get the matrix representation of an operator instance in computational basis
        """
        theta = self._params[0]
        return np.array(
            [
                [1, 0, 0, 0],
                [0, 1, 0, 0],
                [0, 0, cmath.exp(-0.5j * theta), 0],
                [0, 0, 0, cmath.exp(0.5j * theta)],
            ]
        )

    @staticmethod
    def decomposition(lam, qubits, do_queue=False):
        """Decompose this operator into products of other operators.

        Args:
            params (tuple[float, int, array]): operator parameters
            qubits (list[int]): qubits the operator acts on
            do_queue (bool): wether put the decomposed operators into queuing.
        
        Returns:
            list[~.GateBase]
        """
        decomp_ops = [
            PhaseShift(lam / 2, qubits=qubits[1], do_queue=do_queue),
            CNOT(qubits=qubits, do_queue=do_queue),
            PhaseShift(-lam / 2, qubits=qubits[1], do_queue=do_queue),
            CNOT(qubits=qubits, do_queue=do_queue),
        ]
        return decomp_ops

    def adjoint(self, do_queue=False):
        """Get the adjoint operator of this gate operator.

        Adjoint operator tensor is the transposed conjugated tensor of the original gate operator tensor.
        For unitary gate operator, the tensor of adjoint operator is equivalent to the inverse tensor of original gate operator tensor.

        Args:
            do_queue (bool): Optional, default is False. Indicates whether this operator will appear in the quantum circuit.

        Returns:
            GateBase: the adjoint gate

        """
        return CRZ(-self.parameters[0], qubits=self.qubits, do_queue=do_queue)

# TODO: add checking is it really a unitary matrix, see pennylane code
class Unitary(GateBase):
    r"""Unitary(qubits)
    Unitary gate operator.
    The matrix form is:


    **Properties**
        * # of qubits: 1~4
        * # of parameters: 0

    Args:
        qubits (List[int]): The qubits this gate acts on
    """
    _num_params = 0
    _num_qubits = None

    def __init__(self, matrix, qubits: list, do_queue: bool = True, **kwargs):

        self._num_qubits = len(qubits)
        self._user_defined_matrix = matrix

        super().__init__(qubits=qubits, do_queue=do_queue, **kwargs)


    def get_matrix(self):
        r"""Get the matrix representation of an operator instance in computational basis
        """
        return self._user_defined_matrix


    def adjoint(self, do_queue=False):
        """Get the adjoint operator of this gate operator.

        Adjoint operator tensor is the transposed conjugated tensor of the original gate operator tensor.
        For unitary gate operator, the tensor of adjoint operator is equivalent to the inverse tensor of original gate operator tensor.

        Args:
            do_queue (bool): Optional, default is False. Indicates whether this operator will appear in the quantum circuit.

        Returns:
            GateBase: the adjoint gate

        """
        ccj_matrix = deepcopy(self.matrix)
        ccj_matrix = complex_conjugate(ccj_matrix)

        return Unitary(ccj_matrix, qubits=self.qubits, do_queue=do_queue)


def complex_conjugate(ts):
    '''
    obtain complex conjugate
    '''
    shape = ts.shape
    prod_shape = np.prod(shape)
    new_size = np.sqrt(prod_shape).astype(int)
    new_shape = (new_size, new_size)
    ts = ts.reshape(new_shape)
    ts = ts.conj()
    ts = ts.T
    ts = ts.reshape(shape)
    return ts


ops = {
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
    "Rot",
    "PhaseShift",
    "ControlledPhaseShift",
    "CRX",
    "CRY",
    "CRZ",
    "Unitary",
}


obs = {"Hadamard", "PauliX", "PauliY", "PauliZ"}


__all__ = list(ops | obs)
