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

_EINSUM_SYMBOLS_BASE = 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ'

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

        if self.size:
            pop_pos = [i for i, d in enumerate(self.size) if d == 1]
            #print(pop_pos)
            #print(self._size)

            if not pop_pos:  # pylint: disable=no-else-return
                return need_squeeze

            else:
                for pos in pop_pos:
                    # can not use pop() here! since pop will change the content.
                    self._size = [self.size[i] for i, _ in enumerate(self.size) if i != pos]
                    #print("cyc:  ", self.indices[pos])
                    self._indices = [self.indices[i] for i, _ in enumerate(self.indices) if i != pos]
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

    def reindex(self, old_inds, new_inds):

        #self._indices = [new_inds if index == old_inds else index for index in self._indices]
        new_indices = []
        for index in self._indices:
            if index == old_inds:
                new_indices.append(new_inds)
            else:
                new_indices.append(index)
        self._indices = new_indices



    def collapse_repeated_indice(self):
        """Take the diagonals of the repeated index, so that each index
        only appears once.
        """
        # has repeated index
        old_inds = self._indices

        # no repeated index
        new_inds = []
        for index in old_inds:
            if index not in new_inds:
                new_inds.append(index)

        if len(old_inds) > len(new_inds):

            if len(old_inds) > 52:
                raise ValueError(f'Error! This tensor too large for performing einsum')

            down_dict = {}
            for i, scr in enumerate(old_inds):
                down_dict[scr] = i

            einsum_str = ''.join(tuple(_EINSUM_SYMBOLS_BASE[down_dict[scr]] for scr in old_inds))
            einsum_str += '->'
            einsum_str += ''.join(tuple(_EINSUM_SYMBOLS_BASE[down_dict[scr]] for scr in new_inds))
            #print(einsum_str)
            self._data = np.einsum(einsum_str, self._data)
            self._indices = new_inds
            self._size = list(self._data.shape)
            return einsum_str
        return None

    def flip(self, flip_inds):
        """Reverse the axis on this tensor corresponding to ``ind``. Like
        performing e.g. ``X[:, :, ::-1, :]``.
        """
        if flip_inds not in self.indices:
            raise ValueError(f"Can't find index {flip_inds} on this tensor.")

        flipper = []
        for i, index in enumerate(self.indices):
            if index == flip_inds:
                flipper.append(i)


        self._data = np.flip(self._data, flipper)
        return flipper

    def sum_over(self, index):
        dim = self.indices.index(index)
        new_indices = self.indices[:dim] + self.indices[dim + 1:]
        self._indices = new_indices
        self._data = np.sum(self._data, dim)

        return dim


    def select_value(self, index, loc):
        """Select specific values of the index and delete the index. It is like ``M[:,2,:]``.
        Parameters
        ----------
        index : str, index to select specific value and then removed
        loc : int, the position of the specific value to be selected

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

        Examples
        --------
        .. code-block:: python3
            import numpy as np
            data = np.array([[0., 0.,], [1., 2.]])
            indices = ['a', 'b']
            size = [2, 2]
            ts = Tensor(data, indices, size)

            ts.select_value('a', 1)

        >>> ts.data
        array([1.,2.])

        >>> ts.indices
        ['b']

        >>> ts.size
        [2]


        See Also
        --------
        TensorNetwork.select_value
        """

        loc_dict = {index:loc}
        element_selector = tuple(loc_dict.get(ix, slice(None)) for ix in self.indices)
        self._data = self.data[element_selector]

        #print(self.indices, self.indices[0])
        self._indices = [ix for ix in self.indices if ix != index]
        #print(self.indices, index)
        self._size = list(self._data.shape)

        return element_selector


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