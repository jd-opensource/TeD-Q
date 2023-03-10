import numpy as np
from tedq.QInterpreter.operators.ops_abc import GateBase
from .storage_base import CircuitStorage

_EINSUM_SYMBOLS_BASE = 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ'

def circuit_tensor(func, *params):

	with CircuitStorage() as circuit:
		func(*params)

	operators = []
	for ops in circuit.storage_context:
		if isinstance(ops, GateBase):
			operators.append(ops)
		else:
			raise VauleError("circuit_tensor can only receive quantum gate, not initial quantum state or measurement!")

	
	all_qubits = []
	qubits_list = []
	tensors = []
	for op in operators:
		qubits = op._qubits
		tensors.append(op.matrix)
		all_qubits.extend(qubits)
		qubits_list.append(qubits)

	all_qubits = list(set(all_qubits))
	all_qubits.sort()

	new_qubits = [[all_qubits.index(i) for i in qubits] for qubits in qubits_list]
	print(new_qubits)
	num_qubits = len(all_qubits)
	_current_ids = num_qubits - 1
	_layer_ids = list(range(num_qubits))
	_input_indices = []

	for qubits in new_qubits:

		len_qbts = len(qubits)
		_current_ids = _current_ids + len_qbts

		tmpt_input_indices=[]
		for i in reversed(range(len_qbts)):
			tmpt_input_indices.append(_EINSUM_SYMBOLS_BASE[_current_ids - i])
		for i in range(len_qbts):
			tmpt_input_indices.append(_EINSUM_SYMBOLS_BASE[_layer_ids[qubits[i]]])				

		_input_indices.append(
			tmpt_input_indices
			)

		for i in range(len_qbts):
			_layer_ids[qubits[i]] = _current_ids - (len_qbts - 1 - i)


	
	einsum_str = ''
	for indices in _input_indices:
		einsum_str += ''.join(tuple(indices))
		einsum_str += ','

	einsum_str = einsum_str[0:-1]
	einsum_str += '->'
	einsum_str += ''.join(tuple(_EINSUM_SYMBOLS_BASE[i] for i in _layer_ids))
	einsum_str += ''.join(tuple(_EINSUM_SYMBOLS_BASE[i] for i in range(num_qubits)))
	
	print(einsum_str)

	tensors = tuple(tensors)
	print(tensors)
	result = np.einsum(einsum_str, *tensors)

	print(result)
	return result





