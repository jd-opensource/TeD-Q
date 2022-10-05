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
import torch
import toolz
import itertools

from tedq.QInterpreter.operators.measurement import Expectation, Probability, State
from .tensor_core import Tensor
from .array_ops import get_diag_axes, get_antidiag_axes

_EINSUM_SYMBOLS_BASE = 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ'

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
        self.update_tn()

    def simplify(self):
        r'''
        simplify this tensor network
        '''
        print("Wrong! not ready!!! change tn_simplify to False")
        print("Number of tensors BEFORE simplification: ", len(self.tensors))
        self.squeeze()
        self.elliminate_scalar()
        self.fuse_multi_edges(self._output_indices)
        #print("wtf")
        self.diagonal_reduce(self._output_indices)
        self.antidiagonal_reduce(self._output_indices)
        self.weighted_rank_simplify(self._output_indices)
        
        #print(self.operands)

        self.update_tn()

        print("Number of tensors AFTER simplification: ", len(self.tensors))

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

                if do_action == 'pop_scalar':
                    i = order_operand
                    scalar = arrays.pop(i)
                    #print(scalar, arrays[0])
                    arrays[0] = scalar * arrays[0]

                else:
                    raise ValueError("unknown action!!")

            elif len(operand) == 3:
                (do_action, order_operand, action_params) = operand

                if do_action == 'permute':
                    i = order_operand
                    dims = action_params
                    arrays[i] = arrays[i].permute(dims)

                elif do_action == 'reshape':
                    i = order_operand
                    shape = action_params
                    arrays[i] = arrays[i].reshape(shape)

                elif do_action == 'einsum':
                    i = order_operand
                    einsum_str = action_params
                    arrays[i] = torch.einsum(einsum_str, arrays[i])

                elif do_action == 'flip':
                    i = order_operand
                    flipper = action_params
                    array = arrays[i]
                    array = torch.flip(array, flipper)
                    arrays[i] = array

                elif do_action == 'sum_over':
                    i = order_operand
                    dim = action_params
                    array = arrays[i]
                    array = torch.sum(array, dim)
                    arrays[i] = array

                else:
                    raise ValueError("unknown action!!")

            elif len(operand) == 4:

                (do_action, order_operand_i, order_operand_j, action_params) = operand


                if do_action == 'einsum':
                    i, j = order_operand_i, order_operand_j
                    einsum_str = action_params

                    # ATTENTION!!! position j must be found after ta has been pop!!
                    # Since pop(i) will delete ta and then affect position of j!!!
                    # MUST pop i first and then j in order!!
                    ta = arrays.pop(i)
                    tb = arrays.pop(j)

                    new_tensor = torch.einsum(einsum_str, ta, tb)

                    arrays.append(new_tensor)

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
        for t_ids, t in enumerate(self.tensors):  # pylint: disable=invalid-name
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
        for i, t in enumerate(self.tensors):  # pylint: disable=invalid-name
            need_squeeze = t.squeeze()
            #print('need_squeeze:   ', need_squeeze)
            if need_squeeze:
                operand = ('squeeze', i)
                self._operands.append(operand)
        # update the tn
        # self.mapping_tensors() this will not be changed
        self.mapping_indices()
        #return self

    def elliminate_scalar(self):
        queue = list(self.tensors)
        while queue:
            # stop if there only one tensor remained
            if len(self.tensors) <=1 :
                break
            t = queue.pop()
            len_inds = len(t.indices)
            if len_inds == 0:
                i = self.tensors.index(t)
                # delete the scalar from tensor list
                t = self._tensors.pop(i)
                #print(t, self.tensors[0])
                self._tensors[0]._data = t._data * self.tensors[0]._data
                operand = ('pop_scalar', i)
                self._operands.append(operand)

        self.mapping_tensors() # tensors have been changed, re-map them.
        self.mapping_indices()
        self.update_tn()




    def fuse_multi_edges(self, output_inds):
        r'''
        fuse multiple edges into one
        '''
        t_group = collections.defaultdict(list)
        for index, t_ids_set in self._map_index.items():

            # Except output indices
            if index in output_inds:
                continue

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

                    i = self.tensors.index(tensor)

                    operand = ('permute', i, permute)
                    self._operands.append(operand)

                    operand = ('reshape', i, new_size)
                    self._operands.append(operand)
        # update the tn                    
        # self.mapping_tensors() this will not be changed
        self.mapping_indices()
        self.update_tn()

    def diagonal_reduce(
        self,
        output_inds,
        atol=1e-12,
    ):
        """
        Find out diagonal tensors and collapse those axes. This will create 'hyper' edges.
        A--D--B will become    D
                               |
                             A----B  

        Parameters
        ----------
        output_inds : list(str), Outer indices of the tensor network and thus not change. 
        atol : float, optional, The absolute tolerance compared to zero when identifying diagonal tensors.
            which to compare to zero with.
        See Also
        --------
        squeeze fuse_multi_edges
        """

        cache = set()

        queue = list(self._map_tensor)
        #print(len(queue))
        while queue:
            t_ids = queue.pop()
            #print(t_ids)
            t = self._map_tensor[t_ids]

            cache_key = ('dr', t_ids, id(t.data))
            if cache_key in cache:
                continue

            ab = get_diag_axes(t.data, atol=atol)

            # if no diagonal axes
            if ab is None:
                cache.add(cache_key)
                continue

            a, b = ab
            inds_a, inds_b = t.indices[a], t.indices[b]
            if inds_a not in output_inds:
                # transfer inds_a to inds_b
                old_inds = inds_a
                new_inds = inds_b

            else:
                if inds_b in output_inds:
                    # both indices are outer indices, skip them
                    continue
                # make sure output indice inds_a will be kept;
                old_inds = inds_b
                new_inds = inds_a
            #print(t.data)
                

            # transfer indices according to mapping
            # after that, some tensors will have repeated indice
            self.replace_indice(old_inds, new_inds)

            # collapse the repeated indice of this tensor
            # other indice replaced tensors will collapse in their run.
            self.collapse_repeated_indice(t)

            # update the indices mapping
            self.mapping_indices()

            # this tensor might still has other diagonal axes
            queue.append(t_ids)

        # final check of repeated indices
        for _, t in self._map_tensor.items():

            self.collapse_repeated_indice(t)

    
    def antidiagonal_reduce(
        self,
        output_inds,
        atol=1e-12,
    ):
        """
        Find out antidiagonal tensors and collapse those axes. This will create 'hyper' edges.
        A--D--B will become    D
                               |
                             A----B  

        Parameters
        ----------
        output_inds : list(str), Outer indices of the tensor network and thus not change. 
        atol : float, optional, The absolute tolerance compared to zero when identifying antidiagonal tensors.
            which to compare to zero with.
        See Also
        --------
        squeeze fuse_multi_edges
        """

        cache = set()

        queue = list(self._map_tensor)
        while queue:
            t_ids = queue.pop()
            t = self._map_tensor[t_ids]

            cache_key = ('dr', t_ids, id(t.data))
            if cache_key in cache:
                continue

            ab = get_antidiag_axes(t.data, atol=atol)

            # if no diagonal axes
            if ab is None:
                cache.add(cache_key)
                continue

            a, b = ab
            inds_a, inds_b = t.indices[a], t.indices[b]
            if inds_a not in output_inds:
                # transfer flip inds_a
                flip_inds = inds_a

            else:
                if inds_b in output_inds:
                    # both indices are outer indices, skip them
                    continue
                # make sure output indice inds_a will be kept;
                flip_inds = inds_b
                

            #print("flip ", flip_inds)
            # self.operands inside flip function
            self.flip(flip_inds)


            # do diagonal reduce after flipping
            self.diagonal_reduce(output_inds=output_inds)

            # this tensor might still has other diagonal axes
            queue.append(t_ids)


    
    def weighted_rank_simplify(
        self,
        output_inds,
        max_combinations=1000,
    ):
        """Performing contractions of near by tensors that don't increase the weighted rank.
        Parameters

        Parameters
        ----------
        output_inds : list(str), Outer indices of the tensor network and thus not change. 
        max_combinations : int, maximum pair combinations of tensors connected by an index.

        See Also
        --------
        squeeze fuse_multi_edges
        """

        # repeated index checking
        for t in self.tensors:
            self.collapse_repeated_indice(t)

        # scalar checking
        self.elliminate_scalar()

        # update the size dict, will be used in weighting
        self.update_size_dict()

        cache = set()

        counts = collections.Counter()

        # count the freqency of each index
        for t in self.tensors:
            counts.update(t.indices)

        # + 1 for all output indices, so that it will not be pick up to contract
        counts.update(output_inds)


        queue = set(self._map_index)
        while queue:
            index = queue.pop()
            try:
                t_ids_set = self._map_index[index]
            except KeyError:
                continue # index already contracted out in previous process.

            # index only appears in one tensor and it is not an output index, can be summed over
            if counts[index] == 1:
                tid = tids[0]
                t = self._map_tensor[tid]
                dim = t.sum_over(index)

                i = self.tensors.index(t)
                operand = ('sum_over', i, dim)
                self._operands.append(operand)

                # finish this index processing
                continue

            # otherwise try pairwise contractions
            trials = []
            num_combi = 0

            for tid_a, tid_b in itertools.combinations(t_ids_set, 2):

                ta = self._map_tensor[tid_a]
                tb = self._map_tensor[tid_b]

                cache_key = ('rs', tid_a, tid_b, id(ta.data), id(tb.data))
                if cache_key in cache:
                    continue

                num_combi += 1

                # work out the output indices of candidate contraction
                input_indices = []
                input_indices.extend(ta.indices)
                input_indices.extend(tb.indices)
                in_fre = toolz.frequencies(input_indices) # dict of input indices and frequencies

                output_indices = []
                # count of index need to be - 1; A_ab B_ac ->C_abc, count of a reduce by 1
                de_count = []
                for index, c in in_fre.items():
                    if c != counts[index]:
                        output_indices.append(index)
                        if c == 2:
                            de_count.append(index)

                # check if candidate contraction will reduce weighted rank
                wr_a = np.prod([self.size_dict[index] for index in ta.indices])
                wr_b = np.prod([self.size_dict[index] for index in ta.indices])
                wr_o = np.prod([self.size_dict[index] for index in output_indices])

                wr_i = max(wr_a, wr_b)

                if wr_o <= wr_i:
                    candi = (wr_o - wr_i, tid_a, tid_b, output_indices, de_count)
                    trials.append(candi)
                else:
                    cache.add(cache_key)

                # stop if candidate found and reach maximum trials
                if trials and (num_combi > max_combinations):
                    break

            
            if trials:
                # selected the one that reduces the weighted rank most
                candidate = min(trials)
                _, tid_a, tid_b, output_indices, de_count = candidate
                ta = self._map_tensor[tid_a]
                tb = self._map_tensor[tid_b]

                inds_a = ta.indices
                inds_b = tb.indices

                input_inds = set()
                input_inds = input_inds | set(inds_a)
                input_inds = input_inds | set(inds_b)

                if len(input_inds) > 52:
                    raise ValueError(f'Error! This tensor too large for performing einsum')

                down_dict = {}
                for i, scr in enumerate(input_inds):
                    down_dict[scr] = i

                einsum_str = ''.join(tuple(_EINSUM_SYMBOLS_BASE[down_dict[scr]] for scr in inds_a))
                einsum_str += ','
                einsum_str += ''.join(tuple(_EINSUM_SYMBOLS_BASE[down_dict[scr]] for scr in inds_b))
                einsum_str += '->'
                einsum_str += ''.join(tuple(_EINSUM_SYMBOLS_BASE[down_dict[scr]] for scr in output_indices))
                #print(einsum_str)

                data_a = ta.data
                data_b = tb.data
                data_o = np.einsum(einsum_str, data_a, data_b)
                size = [self.size_dict[index] for index in output_indices]

                t_new = Tensor(data_o, output_indices, size)

                # ATTENTION!!! position j must be found after tensor_i has been pop!!
                # Since pop(i) will delete tensor_i and then affect position of j!!!
                # delete old tensors
                i = self.tensors.index(ta)
                self._tensors.pop(i)

                j = self.tensors.index(tb)
                self._tensors.pop(j)

                # add new tensors
                self._tensors.append(t_new)

                # check outputs indices again
                for index in output_indices:
                    queue.add(index)

                operand = ('einsum', i, j, einsum_str)
                self._operands.append(operand)

                # update the tensor network
                self.mapping_tensors()
                self.mapping_indices()


                # decrease the counts
                for index in de_count:
                    counts[index] -= 1

            else:
                # No pairwise contraction found for this index
                continue

        # elliminate scalar tensors produced this simplification
        self.elliminate_scalar()
        self.update_tn()



  


    def flip(self, flip_inds):
        """Flip the dimension corresponding to index ``flip_inds`` on all tensors
        that share it.
        """
        t_ids_set = self._map_index[flip_inds]

        for tid in t_ids_set:
            t = self._map_tensor[tid]
            flipper = t.flip(flip_inds)

            i = self.tensors.index(t)
            assert tid == i
            operand = ('flip', i, flipper)
            self._operands.append(operand)

    def replace_indice(self, old_inds, new_inds):
        """Rename indices of all tensors in the tensor network
        Parameters
        ----------
        old_inds : old index that need to be transferred
        new_inds : new index that to be transferred to
        }
        """

        t_ids_set = self._map_index[old_inds]
        #print(t_ids_set)

        for tid in t_ids_set:
            t = self._map_tensor[tid]
            #print(t.indices)
            t.reindex(old_inds, new_inds)
            #print(t.indices)

    def collapse_repeated_indice(self, t):
        einsum_str = t.collapse_repeated_indice()
        #print(einsum_str)
            
        if einsum_str is not None:
            i = self.tensors.index(t)
            #print("i", i)
            operand = ('einsum', i, einsum_str)
            self._operands.append(operand)



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
        for t in self.tensors:  # pylint: disable=invalid-name
            self._input_indices.append(t.indices)

    def update_size_dict(self):
        r'''
        update the size_dict
        '''
        self._size_dict = {}
        for t in self.tensors:  # pylint: disable=invalid-name
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
                #[get_symbol(_layer_ids[qbts[0]]), get_symbol(_current_ids)]
                [get_symbol(_current_ids), get_symbol(_layer_ids[qbts[0]])]
            )
            _layer_ids[qbts[0]] = _current_ids

        elif len_qbts == 2:
            _current_ids = _current_ids + 2
            _input_indices.append(
                #[get_symbol(_layer_ids[qbts[0]]), get_symbol(
                #_layer_ids[qbts[1]]), get_symbol(_current_ids - 1), get_symbol(_current_ids)]
                [get_symbol(_current_ids - 1), get_symbol(_current_ids), 
                get_symbol(_layer_ids[qbts[0]]), get_symbol(_layer_ids[qbts[1]])]
            )
            _layer_ids[qbts[0]] = _current_ids - 1
            _layer_ids[qbts[1]] = _current_ids

        elif len_qbts == 3:
            _current_ids = _current_ids + 3
            _input_indices.append(
                #[get_symbol(_layer_ids[qbts[0]]), get_symbol(_layer_ids[qbts[1]]), get_symbol(
                #_layer_ids[qbts[2]]), get_symbol(_current_ids - 2), get_symbol(_current_ids - 1), get_symbol(_current_ids)]
                [get_symbol(_current_ids - 2), get_symbol(_current_ids - 1), get_symbol(_current_ids), 
                get_symbol(_layer_ids[qbts[0]]), get_symbol(_layer_ids[qbts[1]]), get_symbol(_layer_ids[qbts[2]])]
            )
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
                        #[get_symbol(_layer_ids_i[qbts[0]]), get_symbol(_current_ids_i)]
                        [get_symbol(_current_ids_i), get_symbol(_layer_ids_i[qbts[0]])]
                        )
                    _layer_ids_i[qbts[0]] = _current_ids_i

                elif len_qbts == 2:
                    _current_ids_i = _current_ids_i + 2
                    _input_indices_i.append(
                        #[get_symbol(_layer_ids_i[qbts[0]]), get_symbol(_layer_ids_i[qbts[1]]), 
                        #get_symbol(_current_ids_i - 1), get_symbol(_current_ids_i)]
                        [get_symbol(_current_ids_i - 1), get_symbol(_current_ids_i), 
                        get_symbol(_layer_ids_i[qbts[0]]), get_symbol(_layer_ids_i[qbts[1]])]
                    )
                    _layer_ids_i[qbts[0]] = _current_ids_i - 1
                    _layer_ids_i[qbts[1]] = _current_ids_i

                elif len_qbts == 3:
                    _current_ids_i = _current_ids_i + 3
                    _input_indices_i.append(
                        #[get_symbol(_layer_ids_i[qbts[0]]), get_symbol(_layer_ids_i[qbts[1]]), 
                        #get_symbol(_layer_ids_i[qbts[2]]), get_symbol(_current_ids_i - 2), 
                        #get_symbol(_current_ids_i - 1), get_symbol(_current_ids_i)]
                        [get_symbol(_current_ids_i - 2), get_symbol(_current_ids_i - 1), 
                        get_symbol(_current_ids_i), get_symbol(_layer_ids_i[qbts[0]]), 
                        get_symbol(_layer_ids_i[qbts[1]]), get_symbol(_layer_ids_i[qbts[2]])]
                    )
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


            # state measurement don't need this
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
        #print(input_indices, output_indices)
        #print(input_arrays, size_dict)
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
