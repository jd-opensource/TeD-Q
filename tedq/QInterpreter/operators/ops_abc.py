#   Copyright 2021-2024 Jingdong Digits Technology Holding Co.,Ltd.

#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at

#       http://www.apache.org/licenses/LICENSE-2.0

#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.

# pylint: disable=line-too-long, trailing-whitespace, too-many-lines, too-many-instance-attributes

r"""
This module contains the abstract base classes for defining tedq quantum gates and observables.

"""

import abc
import itertools
import numpy as np
from tedq.global_variables import GlobalVariables


class OperatorBase(abc.ABC):
    
    r"""
    Abstract Base Class for quantum operators (measurement observables preparation, gates operation).

    Args:
        params (tuple[float]): operator parameters

    Keyword Args:
        qubits (list[int]): Which qubits this operator acts on. Error will occur if not given.
        do_queue (bool): Optional, defaul is True. Indicates whether this operator will appear in the quantum circuit.

    """

    _count = itertools.count() # use to generate sequential and unique instance id.

    def __init__(self, *params, qubits, do_queue = True, **kwargs):

        self._name = self.__class__.__name__  # name of the operator

        if len(params) != self.num_params:
            raise ValueError(
                f'{self._name}: # of parameters is not matched! expected {self.num_params}parameters, but got {len(params)}.')

        self._params = params

        if len(qubits) != self.num_qubits:
            raise ValueError(
                f'{self._name}: # of qubits this operator applied on is not matched! expected {self.num_qubits} qubits, but got {len(qubits)}')

        if do_queue:
            self._instance_id = next(
                self._count
            )  # each time add one; int: index of the Operator in the circuit queue
            _active_warehouse = GlobalVariables.get_value("global_deque")
            if bool(_active_warehouse):
                if bool(_active_warehouse[-1]):
                    _active_warehouse[-1]._append(self)
                else:
                    raise ValueError(
                    "There's no StorageBase for storing information!"
                    )
            else:
                raise ValueError(
                "There's no global_deque for storing information!"
                )

        self._do_queue = do_queue # put this operator in quantum circuit or not
        self._qubits = list(qubits) # list: what qubits the operator applied on
        self._params = list(params) # list: the parameters of this operator
        self._trainable_params = kwargs.pop(
            "trainable_params", list(range(self.num_params))
        )  # what params are trainable, default is all
        self._is_preparation = kwargs.pop("is_preparation", False)
        self._expected_index = kwargs.pop("expected_index", 0) 
        self._matrix = self.get_matrix() # the matrix presentation of this gate data

    def get_matrix(self):
        r"""
        Get the matrix representation of an operator instance in computational basis.

        Returns:
            list[float]: matrix representation.
        """

        raise NotImplementedError

    @property
    def matrix(self):
        r"""
        Matrix representation of an operator instance in computational basis.

        **Example:**

        >>> import tedq as qai
        >>> A = qai.RY(0.5, qubits = [1])
        >>> A.matrix
        array([[ 0.96891242+0.j, -0.24740396+0.j],
                   [ 0.24740396+0.j,  0.96891242+0.j]])

        Returns:
            list[float]: matrix representation.

        """
        
        return self._matrix #generate by get_matrix function

    @property
    def eigvals(self):
        r"""
        Eigenvalues of an operator instance.

        **Example:**

        >>> import tedq as qai
        >>> A = qai.RZ(0.5, qubits = [1])
        >>> A.eigvals
        array([0.96891242-0.24740396j, 0.96891242+0.24740396j])

        Returns:
            array: eigvals representation.
        """

        return np.linalg.eigvals(self.matrix)

    @property
    def num_params(self):
        r""" 
        Get Numbers of paramters.

        Returns:
            int: Number of parameters of this operator.
        """
        return self._num_params  # pylint: disable=no-member


    @property
    def num_qubits(self):
        r""" 
        Get Numbers of qubits.

        Returns:
            int: Number of qubits this operator acts on.
        """
        return self._num_qubits  # pylint: disable=no-member


    @property
    def name(self):
        r"""
        Get the name of the operator.

        Returns:
            String: The name of the operator
        """
        return self._name

    @property
    def qubits(self):
        r"""
        Get the qubits this operator acts on.

        Returns: 
            list[int]: the qubits this operator acts on.
        """
        return self._qubits

    @property
    def parameters(self):
        r"""
        Get parameter values.

        Returns:
            list[float]: List of parameters.
        """
        return self._params

    @property
    def instance_id(self):
        r"""
        A sequencial unique id of this operator instance.

        Returns: 
            int: unique id of the operator.
        """
        if self._do_queue is True:
            return self._instance_id
        print("Caution!!! This operator is not in the queue!")
        return None

    @property
    def trainable_params(self):
        r"""
        Get the list of trainable parameters.

        Returns: 
            list[int]: the list of trainable parameters.
        """
        return self._trainable_params

    @property
    def is_preparation(self):
        r"""
        If the operator used for the state preparation.

        Returns: 
            Bool: If the operator used for the state preparation.
        """
        return self._is_preparation

# =============================================================================
# Subclasses of OperatorBase
# =============================================================================


class GateBase(OperatorBase):
    r"""Base class for quantum gate operator.

    Args:
        params (tuple[float]): operator parameters

    Keyword Args:
        qubits (list[int]): Which qubits this operator acts on. Error will occur if not given.
        do_queue (bool): Optional, defaul is True. Indicates whether this operator will appear in the quantum circuit.
    """


    def __init__(self, *params, qubits: list, do_queue: bool = True, **kwargs):

        self._inverse = False  # wether this gate need to be inversed

        super().__init__(*params, qubits=qubits, do_queue=do_queue, **kwargs)

    @property
    def grad_style(self):
        r"""
        Supported gradient computation style. This function need to be overwrited if it do not support 'param-shift'

        * ``'param-shift'``: Parameter-shift method. Slow but accurate.
        * ``'fint-diff'``: Numerical differentiation using finite difference. Slow but can be used for all gates.
        * ``None``: This quantum gate may not be differentiated.
        """
        #print(self.num_params)
        if self.num_params > 0:  # pylint: disable=no-else-return
            return "param-shift"
        else:
            return None


    def get_parameter_shift(self):
        r"""
        Multiplier and shift for the given parameter.
        By default, differential can be obtain as: :math:`∂f(phi) = m*[f(phi+shift) - f(phi-shift)]`

        Returns: 
            list[list[float]]
        """
        # Default values for multiplier m and center value phi
        shift = np.pi / 2.
        m = 0.5 / np.sin(shift)  # pylint: disable=invalid-name
        phi = 1.
        

        default_param_shift = [[m, phi, shift], [-1.*m, phi, -shift]]
        #print(self.grad_style)
        if self.grad_style == "param-shift":  # pylint: disable=no-else-return
            return default_param_shift
        else:
            raise ValueError(f'{self.name} : This quantum gate do not support parameter-shift method')


    @property
    def is_inverse(self):
        r"""
        If the inverse of the operation was requested.
        
        Returns: 
            bool: The Gate is inverse or not.
        """
        return self._inverse

    def inverse(self):  
        r"""
        Let this gate instace become an inverse gate.
        This function will change the name as well as the matrix of this instance
        
        **Example:**

        >>> import tedq as qai
        >>> A = qai.RZ(0.5, qubits = [1])
        >>> A.name
        RZ
        >>> A.matrix
        array([[0.96891242 - 0.j, 0. + 0.24740396j],
                  [0. + 0.24740396j, 0.96891242 - 0.j]])
        >>> A.inverse()
        >>> A.name
        RZ.inv
        >>> A.matrix
        array([[0.96891242 + 0.j, 0. - 0.24740396j],
                  [0. - 0.24740396j, 0.96891242 + 0.j]])
        """
        if self._inverse:
            self._inverse = False
            self._name = self._name[:-4]

        else:
            self._inverse = True
            self._name = self._name + ".inv"

        # change the martix
        self._matrix = self._matrix.conj().T

        return self

    def adjoint(self, do_queue=False):
        r"""Get the adjoint operator of this gate operator.

        Adjoint operator tensor is the transposed conjugated tensor of the original gate operator tensor.
        For unitary gate operator, the tensor of adjoint operator is equivalent to the inverse tensor of original gate operator tensor.

        Args:
            do_queue (bool): Optional, default is False. Indicates whether this operator will appear in the quantum circuit.

        Returns: 
            GateBase: the adjoint gate

        """
        raise NotImplementedError

# need to be transfer into the version that the order of eigen values is according to the computational basis.
class ObservableBase(OperatorBase):
    """Base class for quantum observables.

    Args:
        params (tuple[float]): operator parameters

    Keyword Args:
        qubits (list[int]): Which qubits this operator acts on. Error will occur if not given.
        do_queue (bool): Optional, defaul is True. Indicates whether this operator will appear in the quantum circuit.
    """


    return_type = None

    @property
    def eigvals(self):
        r"""Eigenvalues of an observable instance.

        **Example:**

        >>> import tedq as qai
        >>> A = qai.PauliZ(qubits = [1])
        >>> A.eigvals
        array([1, -1])

        Returns:
            list[float]: eigvals representation
        """
        return super().eigvals

    def diagonalizing_gates(self):
        r"""Returns the list of gates in which they can
        diagonalize the observable in the computational basis.

        Returns:
            list[~.GateBase]: A list of gates that diagonalize the observable in the computational basis.
        """
        raise NotImplementedError
