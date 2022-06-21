==============================
Backends
==============================
-----------
Overview
-----------
The backend is a function to compute the result of the circuit, as the probability of a specific outcome, the expectation value of the observable, or the final state vector. There're two computation modes to run the quantum simulation -- wave function vector method and tensor network method with optimal contraction path. The computation of the tensor-dot with these two modes is based on the PyTorch and JAX computation modules embedded into the TeD-Q package.

Wave function vector method
==============================
In this mode, the quantum gate acts on the state vector in sequence so that the simulator can acquire the resulting state after each operation of the gate. However, the computation could become a memory- and time-consuming job. With this method, a typical laptop CPU can deal with a simple quantum circuit of up to 20 qubits.

Tensor network method
========================
The simulation of the quantum circuit can be converted into a tensor network. In this method, the sequence of tensor dot operations is reorganized with an optimizer to find the best tensor contraction order so that the computer can calculate the result with less memory and higher speed. The first step of this method is to manipulate the circuit based on the output type and convert the resulting circuit to a hypergraph, which is shown in the later section, and the detailed process of the optimizer is described in the section ``PathOptimizer``. This method is designed for the complex quantum circuit and can be applied to the CPU and GPU clusters. 

Compiled circuit for specific output type
------------------------------------------
	There are three types of output from a circuit.
	
		#. State before measurement
		#. Probability of a specific outcome
		#. The expectation value of the circuit

	The compiled circuit is different for each type of output, so the optimal method to do the tensor contraction is also different. Take the simple two-qubit circuit shown in the previous section; for example, the "state before measurement" circuit is shown in the figure below.

	.. image:: ./assets/img/sample_circuit_cotengra_state.png
		:height: 120
		:alt: Alternative text

	The computer can compute the circuit in the blue box to the state vector :math:`|\phi\psi\rangle.` 

	However, the optimal contraction method to find the state cannot be applied to the circuit of "Probability" or "Expectation value" because the circuit is different. To find the probability of a specific outcome, for instance, qubit #1 to be 0, we need to take the inner product of the measurement operator by the state vector. 
	
	.. math::
		P(A) = \Bigg\langle\phi\psi\Bigg|A\Bigg|\phi\psi\Bigg\rangle

	Therefore, the circuit becomes

	.. image:: ./assets/img/sample_circuit_cotengra_1.png
		:height: 120
		:alt: Alternative text
	
	and 

	.. image:: ./assets/img/sample_circuit_cotengra_2.png
		:height: 120
		:alt: Alternative text

From quantum circuit to hypergraph
-------------------------------------
	The TeD-Q module can convert a quantum circuit to a graph while the quantum gate is the node in the graph. The figure below shows a sample circuit and its corresponding graph. 

	.. image:: ./assets/img/sample_circuit_graph.png
		:height: 120
		:alt: Alternative text

	This graph can be fed into the optimizer to find the optimal contraction sequence. The optimizer in the package includes the well-known `CoTenGra` package and an improved version of it -- `JDOptTN.`

	


----------------------------------
Evaluation of the quantum circuit
----------------------------------

CompiledCircuit
=================

.. automodule:: tedq.backends.compiled_circuit
	:members:
	:private-members:
	:undoc-members:



--------------------
Supported Backends
--------------------

JAX backend
==============
.. automodule:: tedq.backends.jax_backend
	:members:
	:private-members:
	:undoc-members:

PyTorch backend
====================
.. automodule:: tedq.backends.pytorch_backend
	:members:
	:private-members:
	:undoc-members: