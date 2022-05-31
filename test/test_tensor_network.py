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
import numpy as np

from tedq.tensor_network.tensor_network import gen_tensor_networks, TensorNetwork, Tensor


class testGenTensorNetworks():
	r'''
	'''
	pass

class TestTensor():
	r'''
	'''

	def test_construction(self):
		r'''
		'''
		data = np.array([0.1, 0.2])
		indices = ['a']
		size = [2]

		ts = Tensor(data, indices, size)

		assert isinstance(ts, Tensor)

		print("test tensor construction ok!")

	def test_data(self):
		r'''
		'''
		data = np.array([0.1, 0.2])
		indices = ['a']
		size = [2]

		ts = Tensor(data, indices, size)

		judge = ts.data == data
		assert judge.any()

		print("test tensor network data ok!")	

	def test_properties(self):
		r'''
		'''
		data = np.array([0.1, 0.2])
		indices = ['a']
		size = [2]

		ts = Tensor(data, indices, size)

		sz = ts.size
		assert sz == size

		inds = ts.indices
		assert inds == indices

		print("test tensor properties ok!")

	def test_squeeze(self):
		r'''
		'''
		data = np.array([[[1., 0.,], [0., 1.]]])
		indices = ['a', 'b', 'c']
		size = [1, 2, 2]		    
		ts = Tensor(data, indices, size)

		need_squeeze = ts.squeeze()
		assert need_squeeze

		sqz_data = ts.data
		sqz_size = ts.size
		sqz_indices = ts.indices

		judge = sqz_data	== np.array([[1., 0.,], [0., 1.]])
		assert judge.any()
		assert sqz_size == [2, 2]
		assert sqz_indices == ['b', 'c']

		print("test tensor squeeze function ok!")

	def test_fuse(self):
		r'''
		'''
		data = np.array([[1., 0.,], [0., 1.]])
		indices = ['a', 'b']
		size = [2, 2]		    
		ts = Tensor(data, indices, size)

		ts.fuse(['a', 'b'])

		fs_data = ts.data
		fs_indices = ts.indices
		fs_size = ts.size

		judge = fs_data == np.array([1., 0., 0., 1.])
		assert judge.any()
		assert fs_indices == ['a']
		assert fs_size == [4]

		print("test tensor fuse function ok!")


class TestTensorNetwork():
	r'''
	'''

	def test_construction(self):
		r'''
		'''
		input_arrays = [np.array([0.1, 0.2]), np.array([[0.3, 0.4], [0.3, 0.4]])]
		input_indices = [['a'], ['a', 'b']]
		output_indices = ['b']
		size_dict = {'a':2, 'b':2}

		tn = TensorNetwork(input_arrays, input_indices, output_indices, size_dict)

		assert isinstance(tn, TensorNetwork)

		tensors = tn.tensors
		for ts in tensors:
			assert isinstance(ts, Tensor)

		print("test tensor network construction ok!")

	def test_data(self):
		r'''
		'''
		input_arrays = [np.array([0.1, 0.2]), np.array([[0.3, 0.4], [0.3, 0.4]])]
		input_indices = [['a'], ['a', 'b']]
		output_indices = ['b']
		size_dict = {'a':2, 'b':2}

		tn = TensorNetwork(input_arrays, input_indices, output_indices, size_dict)

		tensors = tn.tensors
		for i, ts in enumerate(tensors):
			judge = ts.data == input_arrays[i]
			assert judge.any()

		print("test tensor network data ok!")	

	def test_properties(self):
		r'''
		'''
		input_arrays = [np.array([0.1, 0.2]), np.array([[0.3, 0.4], [0.3, 0.4]])]
		input_indices = [['a'], ['a', 'b']]
		output_indices = ['b']
		size_dict = {'a':2, 'b':2}

		tn = TensorNetwork(input_arrays, input_indices, output_indices, size_dict)

		s_d = tn.size_dict
		assert size_dict == s_d

		in_inds = tn.input_indices
		assert input_indices == in_inds

		out_inds = tn.output_indices
		assert output_indices == out_inds

		print("test tensor network properties ok!")		