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


#TODO: add tests for cotengra and cyc_cotengra
r"""
Test file for storage and circuit.
"""

from test_visualization import TestVisulization
from test_operators import TestGateOperators, TestObservableOperators, TestMeasurement
from test_circuit import Test_CircuitStorage, test_Circuit, test_Circuit_io
from test_compiled_circuit import Test_compiled_circuit
from test_jax_backend import Test_jax_backend
from test_pytorch_backend import Test_pytorch_backend
from test_optimizers import Test_gradient_descent, Test_pytorch_built_in_optimizer
from test_ray import Test_RayParallel
from test_tensor_network import testGenTensorNetworks, TestTensorNetwork, TestTensor

def run_the_test():
	"""
	"""
	# visualization
	print(" ")
	print("Star testing visualization function!")
	print(" ")
	a = TestVisulization()
	a.test_initialization()
	print(" ")
	print("Finish visualization module test, everything ok!")
	print(" ")

	# gate operators
	print(" ")
	print("Star testing gate operators function!")
	print(" ")
	a = TestGateOperators()
	a.test_construction()
	a.test_incorrect_num_qubits()
	a.test_incorrect_num_params()
	a.test_no_qubits_args_passed()
	a.test_adjoint()
	a.test_inversion()
	a.test_trainable_parameters()
	print(" ")
	print("Finish gate operators module test, everything ok!")
	print(" ")

	# observable operators
	print(" ")
	print("Star testing observable operators function!")
	print(" ")
	a = TestObservableOperators()
	a.test_construction()
	print(" ")
	print("Finish observable operators module test, everything ok!")
	print(" ")

	# measurement operators
	print(" ")
	print("Star testing measurement function!")
	print(" ")
	a = TestMeasurement()
	a.test_construction()
	a.test_return_type()
	a.test_wrong_qubits_obs_input()
	print(" ")
	print("Finish measurement module test, everything ok!")
	print(" ")

	# circuit storage
	print(" ")
	print("Star testing circuit storage function!")
	print(" ")
	a = Test_CircuitStorage()
	a.test_construction()
	a.test_recording()
	a.test_append()
	a.test_remove()
	print(" ")
	print("Finish circuit storage module test, everything ok!")
	print(" ")

	# circuit storage
	print(" ")
	print("Star testing circuit function!")
	print(" ")
	a = test_Circuit()
	a.test_construction()
	a.test_incorrect_num_qubits()
	a.test_unknow_content_in_queue()
	a.test_gate_inside_circuit()
	a.test_measurement_inside_circuit()
	print(" ")
	print("Finish circuit module test, everything ok!")
	print(" ")

	# circuit io
	print(" ")
	print("Star testing circuit io!")
	print(" ")
	a = test_Circuit_io()
	a.test_reading_qsim_file()
	print(" ")
	print("Finish circuit_io module test, everything ok!")
	print(" ")

	# compiled circuit
	print(" ")
	print("Star testing compiled circuit function!")
	print(" ")
	a = Test_compiled_circuit()
	a.test_construction()
	a.test_operators_measuremts()
	a.test_confliction_order_finder()
	a.test_SVPM_axes_list_and_perms_list()
	a.test_circuit_to_tensor_network_convertion()
	a.test_trainable_parameters()
	a.test_update_parameters()
	print(" ")
	print("Finish compiled circuit module test, everything ok!")
	print(" ")

	# jax backend
	print(" ")
	print("Star testing jax backend function!")
	print(" ")
	a = Test_jax_backend()
	a.test_construction()
	a.test_unknown_interface()
	a.test_confliction_order_finder()
	a.test_tensor_data()
	a.test_init_state()
	a.test_SVPM_forward_jax_interface()
	a.test_SVPM_forward_pytorch_interface()
	a.test_SVPM_backward_jax_interface()
	a.test_SVPM_backward_pytorch_interface()
	print(" ")
	print("Finish jax backend module test, everything ok!")
	print(" ")

	# pytorch backend
	print(" ")
	print("Star testing pytorch backend function!")
	print(" ")
	a = Test_pytorch_backend()
	a.test_construction()
	a.test_unknown_interface()
	a.test_confliction_order_finder()
	a.test_diff_input_dtype_param_shift_method()
	a.test_tensor_data()
	a.test_init_state()
	a.test_SVPM_forward()
	a.test_SVPM_backward_back_prop()
	a.test_SVPM_backward_param_shift()
	print(" ")
	print("Finish pytorch backend module test, everything ok!")
	print(" ")

	# gradient descent optimizer
	print(" ")
	print("Star testing gradient descent optimizer pytorch backend function!")
	print(" ")
	a = Test_gradient_descent()
	a.test_construction()
	a.test_change_step_size()
	a.test_with_qubit_rotation_example()
	print(" ")
	print("Finish gradient descent optimizer module test, everything ok!")
	print(" ")

	# pytorch built-in optimizer
	print(" ")
	print("Star testing pytorch built-in optimizer compatability with tedq pytorch backend function!")
	print(" ")
	a = Test_pytorch_built_in_optimizer()
	a.test_with_qubit_rotation_example()
	print(" ")
	print("Finish pytorch built-in optimizer compatability test, everything ok!")
	print(" ")
	print(" ")

	
	print(" ")
	print("Star testing ray parallel function!")
	print(" ")
	a = Test_RayParallel()
	a.test_ray_initialization()
	a.test_ray_shutdown()
	print(" ")
	print("Finish ray parallel function test, everything ok!")
	print(" ")
	print(" ")

	print(" ")
	print("Star testing tensor function!")
	print(" ")
	a = TestTensor()
	a.test_construction()
	a.test_data()
	a.test_properties()
	a.test_squeeze()
	a.test_fuse()
	print(" ")
	print("Finish tensor function test, everything ok!")
	print(" ")
	print(" ")

	print(" ")
	print("Star testing tensor network function!")
	print(" ")
	a = TestTensorNetwork()
	a.test_construction()
	a.test_data()
	a.test_properties()
	print(" ")
	print("Finish tensor network function test, everything ok!")
	print(" ")
	print(" ")


	print("All the tests pass! congratulation! Now you can play with TeD-Q, good luck~!")