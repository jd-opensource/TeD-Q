import tedq as qai
import numpy as np
import torch

class Template:
	r"""
	This is the base class for the template circuit.
	"""
	def __init__(self):
		pass
	def __str__(self): 
		return "Template Layer: "+self.__class__.__name__
	


class RandomLayer(Template):
	r"""
	This is the template randomly generates the quantum circuit with specific depth. The circuit are consist of single-qubit rotation gates and two-qubit controlled-NOT gates.
	
	Args:
		n_wires(int):	Numbers of the qubits in the circuit
		n_layers(int):	Depth of the randomly-generated circuit
		ratio(int):		Ratio between number of single-qubit gates and two-qubits gates. The value is between 0 and 1.
		rand_params:	Parameters for the rotation gates.

	**Example**

	.. code-block:: python3

		np.random.seed(21)
		n_layers = 4
		n_qubits = 4
		rand_params = np.random.uniform(high=2 * np.pi, size=(n_layers, n_qubits))
		def circuitDef():
			qai.Templates.RandomLayer(n_qubits, n_layers, 0.3, rand_params)
			exp_vals = [qai.measurement.expval(qai.PauliZ(qubits=[position])) for position in range(n_qubits)]
		#compile the quantum circuit
		circuit = qai.Circuit(circuitDef, n_qubits)
		# Draw the circuit
		drawer = qai.matplotlib_drawer(circuit)
		drawer.full_draw()
	
	.. image:: ./assets/img/template_sample_random_layer.png
		:height: 120
		:alt: Sample circuit of random layer.

	"""
	def __init__(self, n_wires, n_layers, ratio, rand_params):
		super().__init__()
		
		rotations = [qai.RX, qai.RY, qai.RZ]
		for i in range(n_layers):
			j=0
			while j < (np.shape(rand_params)[1]):
				if np.random.random()<ratio: #cnot
					rnd_wires = np.random.choice(range(n_wires),2, replace=False)
					qai.CNOT(qubits=rnd_wires, trainable_params=[])
					
				else: #rotation
					gate = np.random.choice(rotations)
					rnd_wire = np.random.randint(n_wires)
					gate(torch.tensor(rand_params[i][j]), qubits=[rnd_wire], trainable_params=[])
					j+=1

class HardwareEfficient(Template):
	r"""
	This is the hardware-efficient quantum circuit with specific depth. In each layer, the neighbor qubits are connected with a controlled-NOT gate.
	
	Args:
		n_wires(int):	Numbers of the qubits in the circuit
		n_layers(int):	Depth of the randomly-generated circuit
		params:	Parameters for the rotation gates.

	**Example**

	.. code-block:: python3

		np.random.seed(21)
		n_layers = 4
		n_qubits = 4
		rand_params = np.random.uniform(high=2 * np.pi, size=(n_layers*2+2, n_qubits))
		def circuitDef():
		    qai.Templates.HardwareEfficient(n_qubits, n_layers, rand_params)
		    exp_vals = [qai.measurement.expval(qai.PauliZ(qubits=[position])) for position in range(n_qubits)]
		#compile the quantum circuit
		circuit = qai.Circuit(circuitDef, n_qubits)
		# Draw the circuit
		drawer = qai.matplotlib_drawer(circuit)
		drawer.full_draw()
	
	.. image:: ./assets/img/template_sample_fully_connected.png
		:height: 120
		:alt: Sample circuit of random layer.

	"""
	def __init__(self, n_wires, depth, params):
		super().__init__()
		for i in range(depth):

			for j in range(n_wires):
				qai.RY(torch.tensor(params[2*i][j]), qubits=[j])
				qai.RZ(torch.tensor(params[2*i+1][j]), qubits=[j])

			idx = 2
			while idx<n_wires:
				qai.CNOT(qubits=(idx-1, idx))
				idx+=2
			idx = 1
			while idx<n_wires:
				qai.CNOT(qubits=(idx-1, idx))
				idx+=2
			
		for j in range(n_wires):
			qai.RY(torch.tensor(params[2*depth][j]), qubits=[j])
			qai.RZ(torch.tensor(params[2*depth+1][j]), qubits=[j])

class FullyConnected(Template):
	r"""
	This is the fully-connected quantum circuit with specific depth. In each layer, the all qubits are connected with each other via a series of controlled-NOT gate.
	
	Args:
		n_wires(int):	Numbers of the qubits in the circuit
		n_layers(int):	Depth of the randomly-generated circuit
		params:	Parameters for the rotation gates.

	**Example**

	.. code-block:: python3

		np.random.seed(21)
		n_layers = 2
		n_qubits = 4
		rand_params = np.random.uniform(high=2 * np.pi, size=(n_layers*2+2, n_qubits))
		def circuitDef():
		    qai.Templates.FullyConnected(n_qubits, n_layers, rand_params)
		    exp_vals = [qai.measurement.expval(qai.PauliZ(qubits=[position])) for position in range(n_qubits)]
		#compile the quantum circuit
		circuit = qai.Circuit(circuitDef, n_qubits)
		# Draw the circuit
		drawer = qai.matplotlib_drawer(circuit)
		drawer.full_draw()
	
	.. image:: ./assets/img/template_sample_hardware_efficient.png
		:height: 120
		:alt: Sample circuit of random layer.

	"""
	def __init__(self, n_wires, depth, params):
		super().__init__()
		for i in range(depth):

			for j in range(n_wires):
				qai.RY(torch.tensor(params[2*i][j]), qubits=[j])
				qai.RZ(torch.tensor(params[2*i+1][j]), qubits=[j])

			idx = 2
			for j in range(n_wires-1, -1, -1):
				for k in range(j+1, n_wires):
					qai.CNOT(qubits=(j, k))
			
			
		for j in range(n_wires):
			qai.RY(torch.tensor(params[2*depth][j]), qubits=[j])
			qai.RZ(torch.tensor(params[2*depth+1][j]), qubits=[j])
