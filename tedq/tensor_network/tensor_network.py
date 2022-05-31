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
This module contains the :class:`TensorNetwork` and :class:`Tensor`.
"""

# pylint: disable=line-too-long, trailing-whitespace, too-many-lines, too-many-instance-attributes, too-few-public-methods, too-many-locals


import collections
from copy import deepcopy
import numpy as np

from tedq.QInterpreter.operators.measurement import Expectation, Probability, State

class Tensor():
    r'''
    Stroe tensor information, including data, indices and size.
    '''

    def __init__(self, data, indices, size):
        self._data = data
        self._indices = indices
        self._size = size
        size_from_data = list(self._data.shape)
        if size_from_data != self._size:
            raise ValueError("Error!!, input size do not match data shape!!")

    def squeeze(self):
        """Drop any singlet dimensions from this tensor.
        Parameters
        ----------
        -------
        Tensor

        **Example**

        .. code-block:: python3

            import numpy as np
            data = np.array([[[1., 0.,], [0., 1.]]])
            indices = ['a', 'b', 'c']
            size = [1, 2, 2]            

            ts = Tensor(data, indices, size)

        >>> ts.data
        array([[[1., 0.],
                [0., 1.]]])

        >>> ts.indices
        ['a', 'b', 'c']

        >>> ts.size
        [1, 2, 2]

        >>> ts.squeeze()

        >>> ts.data
        array([[1., 0.],
                [0., 1.]])

        >>> ts.indices
        ['b', 'c']

        >>> ts.size
        [2, 2]

        """
        #pass
        
        need_squeeze = False

        if self._size:
            pop_pos = [i for i, d in enumerate(self._size) if d == 1]
            #print(pop_pos)
            #print(self._size)

            if not pop_pos:  # pylint: disable=no-else-return
                return need_squeeze

            else:
                for pos in pop_pos:
                    self._size.pop(pos)
                    self._indices.pop(pos)
                    self._data = self._data.squeeze()

                size_from_data = list(self._data.shape)
                if size_from_data != self._size:
                    raise ValueError("Error!!, tensor size do not match data shape!!")
                need_squeeze = True
                return need_squeeze

        return need_squeeze



    def fuse(self, fused_inds):
        r'''
        Fuse multiple indices into one.

        **Example**

        .. code-block:: python3

            import numpy as np
            data = np.array([[1., 0.,], [0., 1.]])
            indices = ['a', 'b']
            size = [2, 2]            

            ts = Tensor(data, indices, size)

        >>> ts.data
        array([[1., 0.],
                [0., 1.]])

        >>> ts.indices
        ['a', 'b']

        >>> ts.size
        [2, 2]

        >>> ts.fuse(['a', 'b'])

        >>> ts.data
        array([1., 0., 0., 1.])

        >>> ts.indices
        ['a']

        >>> ts.size
        [4]

        '''

        num_fused = len(fused_inds)
        fused_set = set(fused_inds)
        #remain_inds = fused_inds[0]
        #fused = fused_inds[1:]
        indices = self.indices

        old_set = set(indices)
        wrong = fused_set - old_set
        if wrong:
            raise ValueError("Error! fused_inds has indices that not appear in tersor's original indices")

        new_indices = list(fused_inds)
        #print(new_indices)
        #print(fused_inds)

        for index in indices:
            if not index in fused_set:
                new_indices.append(index)

        len_old = len(indices)
        len_new = len(new_indices)
        if len_old != len_new:
            #print(indices)
            #print(fused_set)
            #print(new_indices)
            raise ValueError("lenth of new indices is not equal to old one!!")

        permute = [indices.index(index) for index in new_indices]
        size = self.size

        #print(num_fused)
        #print(permute)
        #print(size)
        #print(tuple(size[permute[i]] for i in range(num_fused)))
        size_prod = np.prod(tuple(size[permute[i]] for i in range(num_fused)))
        #print(size_prod)

        l_new_size = [size_prod]
        #print(l_new_size)

        num_size = len(size)
        r_new_size = [size[permute[i]] for i in range(num_fused, num_size)]

        new_size = l_new_size + r_new_size# can not use extend, otherwise new_size is None, l_new_size will be extended!!!

        #print(new_indices)
        #print(indices)
        #print(permute)
        #print(size)
        #print(l_new_size)
        #print(r_new_size)
        #print(new_size)
        #print(" ")
        self._data = np.transpose(self._data, axes = permute)
        self._data = self._data.reshape(new_size)
        self._size = new_size
        self._indices = [new_indices[0]] + new_indices[num_fused:]
        #print(self._indices)

        return (permute, new_size)





    @property
    def data(self):
        r'''
        return data of the tensor
        '''
        return self._data

    @property
    def indices(self):
        r'''
        return indices of the tensor
        '''
        return self._indices

    @property
    def size(self):
        r'''
        return size of the tensor
        '''
        return self._size

class TensorNetwork():
    r'''
    A quantum circuit can be transfered into a tensor network, and this tensor network can be simplified.
    '''

    def __init__(self, input_arrays, input_indices, output_indices, size_dict):

        self._input_indices = input_indices
        self._output_indices = output_indices
        self._size_dict = size_dict
        self._tensors = []
        self._map_tensor = {}
        self._map_index = collections.defaultdict(set)
        self._operands = []
        # checking
        len_arrays = len(input_arrays)
        len_indices = len(self._input_indices)
        if len_arrays != len_indices:
            print(len(input_arrays))
            print(len(self._input_indices))
            raise ValueError(f'Error! number of input arrays {len_arrays} and input indices {len_indices} do not match!!')

        # convert input indices into set
        indices_set = set()
        for indices in self._input_indices:
            tmpt = set(indices)
            indices_set = indices_set.union(tmpt)

        # convert output indices into set
        tmpt = set(self._output_indices)    
        indices_set = indices_set.union(tmpt)

        # convert size dict indices into set
        size_dict_set = set(self._size_dict.keys())

        # difference between sets
        diff_set = indices_set^size_dict_set

        # checking
        if len(diff_set):
            raise ValueError("Error! indices of input do not match with size_dict!!")

        for i, _ in enumerate(input_arrays):
            data = input_arrays[i]
            indices = self._input_indices[i]
            size = [self._size_dict[index] for index in indices]

            tensor = Tensor(data, indices, size)
            self._tensors.append(tensor)

        self.mapping_tensors()
        self.mapping_indices()

    def simplify(self):
        r'''
        simplify this tensor network
        '''
        print("Wrong! not ready!!! change tn_simplify to False")
        self.squeeze()
        self.fuse_multi_edges()
        

        #print(self.operands)

        self.update_tn()

    def simplify_arrays(self, arrays):
        r'''
        According to the tensor network simplification, simplified the input arrays.
        '''
        print("Wrong! not ready!!! change tn_simplify to False")
        for operand in self.operands:
            if len(operand) == 2:
                (do_action, order_operand) = operand
                if do_action == 'squeeze':
                    i = order_operand
                    arrays[i] = arrays[i].squeeze()
                else:
                    raise ValueError("unknown action!!")

            elif len(operand) == 3:
                (do_action, order_operand, action_params) = operand

                if do_action == 'permute':
                    i = order_operand
                    arrays[i] = arrays[i].permute(action_params)

                elif do_action == 'reshape':
                    i = order_operand
                    arrays[i] = arrays[i].reshape(action_params)

                else:
                    raise ValueError("unknown action!!")

            else:
                print(len(operand))
                raise ValueError("Lenght of operand is not right!!")

        return arrays



    def mapping_tensors(self):
        r'''
        mapping tensor
        '''
        self._map_tensor = {}
        for t_ids, t in enumerate(self._tensors):  # pylint: disable=invalid-name
            self._map_tensor[t_ids] = t

    def mapping_indices(self):
        r'''
        mapping indices
        '''
        self._map_index = collections.defaultdict(set)
        for t_ids, t in self._map_tensor.items():  # pylint: disable=invalid-name
            indices = t.indices
            for index in indices:
                self._map_index[index].add(t_ids)

    def squeeze(self):
        r'''
        squeeze all the tensor inside the tensor network
        '''
        for i, t in enumerate(self._tensors):  # pylint: disable=invalid-name
            need_squeeze = t.squeeze()
            #print('need_squeeze:   ', need_squeeze)
            if need_squeeze:
                operand = ('squeeze', i)
                self._operands.append(operand)
        # update the tn
        self.mapping_tensors()
        self.mapping_indices()
        #return self

    def fuse_multi_edges(self):
        r'''
        fuse multiple edges into one
        '''
        t_group = collections.defaultdict(list)
        for index, t_ids_set in self._map_index.items():
            length = len(t_ids_set)
            if length > 1:
                ids = frozenset(t_ids_set)
                t_group[ids].append(index)

        for t_set, indices in t_group.items():
            length = len(indices)
            if length > 1:
                for t_ids in t_set:
                    tensor = self._map_tensor[t_ids]
                    (permute, new_size) = tensor.fuse(indices)

                    i = self._tensors.index(tensor)

                    operand = ('permute', i, permute)
                    self._operands.append(operand)

                    operand = ('reshape', i, new_size)
                    self._operands.append(operand)
        # update the tn                    
        self.mapping_tensors()
        self.mapping_indices()


    def update_tn(self):
        r'''
        update the tensor network
        '''
        self.update_indices()
        self.update_size_dict()

    def update_indices(self):
        r'''
        update the indices.
        '''
        self._input_indices = []
        for t in self._tensors:  # pylint: disable=invalid-name
            self._input_indices.append(t.indices)

    def update_size_dict(self):
        r'''
        update the size_dict
        '''
        self._size_dict = {}
        for t in self._tensors:  # pylint: disable=invalid-name
            for indice, size in zip(t.indices, t.size):
                old_size = self._size_dict.get(indice, None)# old version: get(indice, default=None), but now don't need keyword 'default'
                if old_size:
                    if old_size != size:
                        raise ValueError("Error!!, indice in different tensors has different size!!")
                else:
                    self._size_dict[indice] = size

    @property
    def input_indices(self):
        r'''
        return the indices of input array
        '''
        return self._input_indices

    @property
    def output_indices(self):
        r'''
        return the indices of output array
        '''
        return self._output_indices

    @property
    def size_dict(self):
        r'''
        return the size_dict
        '''
        return self._size_dict

    @property
    def tensors(self):
        r'''
        return the tensors
        '''
        return self._tensors

    @property
    def operands(self):
        r'''
        return the operators that can simplified the tensor network.
        '''
        return self._operands

    def clean_operands(self):
        r'''
        clean the operators that can simplified the tensor network.
        '''
        self._operands = []






# pylint: disable=too-many-branches, too-many-statements
def gen_tensor_networks(num_qubits, operators, appliedqubits, measurements):
    r'''
    function that generate tensor networks according to input quantum circuit parameters
    '''

    tn_operands = []  # each measurement will have its own operand
    _layer_ids = list(range(num_qubits))
    _current_ids = num_qubits - 1
    _input_indices = []
    _input_arrays = []
    for i in _layer_ids:
        _input_indices.append(list(get_symbol(i)))
        _input_arrays.append(np.array([1., 0.]))

    #print(len(_input_arrays))
    #print(len(_input_indices))

    for idx, qbts in appliedqubits.items():  # idx qubits

        len_qbts = len(qbts)

        if len_qbts == 1:
            _current_ids = _current_ids + 1
            _input_indices.append(
                [get_symbol(_layer_ids[qbts[0]]), get_symbol(_current_ids)])
            _layer_ids[qbts[0]] = _current_ids

        elif len_qbts == 2:
            _current_ids = _current_ids + 2
            _input_indices.append([get_symbol(_layer_ids[qbts[0]]), get_symbol(
                _layer_ids[qbts[1]]), get_symbol(_current_ids - 1), get_symbol(_current_ids)])
            _layer_ids[qbts[0]] = _current_ids - 1
            _layer_ids[qbts[1]] = _current_ids

        elif len_qbts == 3:
            _current_ids = _current_ids + 3
            _input_indices.append([get_symbol(_layer_ids[qbts[0]]), get_symbol(_layer_ids[qbts[1]]), get_symbol(
                _layer_ids[qbts[2]]), get_symbol(_current_ids - 2), get_symbol(_current_ids - 1), get_symbol(_current_ids)])
            _layer_ids[qbts[0]] = _current_ids - 2
            _layer_ids[qbts[1]] = _current_ids - 1
            _layer_ids[qbts[2]] = _current_ids

        else:
            if len_qbts == 12:
                _egg = []
                for cc in range(len_qbts):
                    _symbol = get_symbol(qbts[cc])
                    _egg.append(_symbol)
                _s_egg = ''.join(_egg)
                raise ValueError(f'{_s_egg} remind you, please change tn_simplify value!')
            raise ValueError("Error!! unknown operator with len of applied qubits larger than 3!")

        operator = operators[idx]
        matrix = operator.matrix
        operator_num_qubits = operator.num_qubits
        matrix_shape = [2 for _ in range(int(2*operator_num_qubits))]
        _input_arrays.append(matrix.reshape(matrix_shape))

    #print(len(_input_arrays))
    #print(len(_input_indices))

    for measurement in measurements:

        _output_indices_i = []

        _current_ids_i = deepcopy(_current_ids)
        _input_indices_i = deepcopy(_input_indices)
        _layer_ids_i = deepcopy(_layer_ids)

        _input_arrays_i = deepcopy(_input_arrays)

        #print(len(_input_arrays_i))
        #print(len(_input_indices_i))

        if measurement.return_type is State:
            _output_indices_i = list(get_symbol(
                _layer_ids_i[i]) for i in range(num_qubits))

        else:

            if measurement.return_type is Expectation:
                _current_ids_i = _current_ids_i + 1
                layer = measurement.obs.qubits[0]
                _input_indices_i.append(
                    [get_symbol(_layer_ids_i[layer]), get_symbol(_current_ids_i)])
                _layer_ids_i[layer] = _current_ids_i
                _input_arrays_i.append(measurement.obs.matrix)

            if measurement.return_type is Probability:
                if measurement.qubits is None:
                    pass
                else:
                    _output_indices_i = list(get_symbol(
                        _layer_ids_i[qbts]) for qbts in measurement.qubits)

            #print(len(_input_arrays_i))
            #print(len(_input_indices_i))

            for idx, qbts in reversed(appliedqubits.items()):  # idx qubits

                len_qbts = len(qbts)

                if len_qbts == 1:
                    _current_ids_i = _current_ids_i + 1
                    _input_indices_i.append(
                        [get_symbol(_layer_ids_i[qbts[0]]), get_symbol(_current_ids_i)])
                    _layer_ids_i[qbts[0]] = _current_ids_i

                elif len_qbts == 2:
                    _current_ids_i = _current_ids_i + 2
                    _input_indices_i.append([get_symbol(_layer_ids_i[qbts[0]]), get_symbol(
                        _layer_ids_i[qbts[1]]), get_symbol(_current_ids_i - 1), get_symbol(_current_ids_i)])
                    _layer_ids_i[qbts[0]] = _current_ids_i - 1
                    _layer_ids_i[qbts[1]] = _current_ids_i

                elif len_qbts == 3:
                    _current_ids_i = _current_ids_i + 3
                    _input_indices_i.append([get_symbol(_layer_ids_i[qbts[0]]), get_symbol(_layer_ids_i[qbts[1]]), get_symbol(
                        _layer_ids_i[qbts[2]]), get_symbol(_current_ids_i - 2), get_symbol(_current_ids_i - 1), get_symbol(_current_ids_i)])
                    _layer_ids_i[qbts[0]] = _current_ids_i - 2
                    _layer_ids_i[qbts[1]] = _current_ids_i - 1
                    _layer_ids_i[qbts[2]] = _current_ids_i
        

                else:
                    raise ValueError("Error!! unknown operator with len of applied qubits larger than 3!")

                operator = operators[idx].adjoint()
                matrix = operator.matrix
                operator_num_qubits = operator.num_qubits
                matrix_shape = [2 for _ in range(int(2*operator_num_qubits))]
                _input_arrays_i.append(operators[idx].matrix.reshape(matrix_shape))


        for i in _layer_ids_i:
            _input_indices_i.append(list(get_symbol(i)))
            _input_arrays_i.append(np.array([1., 0.]))

        _size_dict_i = {}

        for ids in _input_indices_i:
            for symbol in ids:
                _size_dict_i[symbol] = 2
        for symbol in _output_indices_i:
            _size_dict_i[symbol] = 2

        tn_operands.append(
            (_input_arrays_i, _input_indices_i, _output_indices_i, _size_dict_i))

    tensor_networks = []
    for operand in tn_operands:
        (input_arrays, input_indices, output_indices, size_dict) = operand
        tensor_network = TensorNetwork(input_arrays, input_indices, output_indices, size_dict)
        tensor_networks.append(tensor_network)

    return tensor_networks









def get_symbol(i):
    """Get the symbol corresponding to int ``i`` - runs through the usual 52
    letters before resorting to unicode characters, starting at ``chr(192)``.   

    Examples:
    
    >>> get_symbol(1)
    'b' 

    >>> get_symbol(200)
    'Ŕ' 

    >>> get_symbol(20000)
    '京'
    """
    _einsum_symbols_base = 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ'
    if i < 52:
        return _einsum_symbols_base[i]
    return chr(i + 140)
