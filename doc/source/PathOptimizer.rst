--------------------------------
Contraction path optimizer
--------------------------------

The contraction path optimizer in the TeD-Q allows users to speed up the tensor operation in quantum simulation. Finding the optimal contraction path reduces the complexity of calculating the tensor network compared to computing it in sequence. In the TeD-Q, users can use the build-in Jingdong Optimizer for Tensor Network (JDOptTN) or CoTenGra to optimize the contraction path.

Overview of Jingdong Optimizer for Tensor Network
=====================================================

In the TeD-Q, the built-in Jingdong Optimizer for Tensor Network (JDOptTN) and CotenGra can serve as the contraction path optimizer. The following describes the overall workflow.

	#. Abstraction of tensor network
	#. Generate trials of contraction path
	#. Slice the trial path
	#. Find the trial path with minimal computing cost
	#. Compute the tensor network along the path

In the following section, we use the single-core scenario to describe the function of each step.

Abstraction of tensor network
------------------------------
In this step, the tensor network of quantum circuit are converted to a list of indices of tensors. The information about the connection between tensor will be preserved, for example, 

.. math::
		T_{ab}T_{bc} \rightarrow [[a, b], [b, c]].

Together with the size of each indice, 

.. math::
		[a: 2, b: 2, c: 2]

These two pieces of information can describe the connection between tensors without using the complete data of the tensors.

Generate trials of contraction path
------------------------------------
The JDOptTN module will feed the abstracted tensor network into the partition function to separate the network into two pieces. The partition is based on the KaHyPar partition module, aimed at finding the edge with minimal connection, that is, minimal computation complexity. To prevent bias from a single partition setting, the threshold of imbalance is random so that we can generate different trial paths every time. The division will recursively run until only two tensors are remaining. The partition result will be stored in the binary tree structure so that the DFS path from the leaves to the root is the path to calculate the tensor network. 

Slice the trial contraction path
---------------------------------
The memory requirement becomes massive for the tensor network of a many-qubit circuit. To reduce the memory requirement, we can use the slice function to minimize the need for memory during the tensor network computation. Take the following tensor product as an example:

.. math::
		A_{mn}B_{np} \rightarrow C_{mp}.

The required amount of memory is proportional to 

.. math::
		m\times p\times (2\times 2-1)

After applying the slice function, the computation of the tensor product is split into two uncomplicated parts with the indices n fixed.

.. math::
		A_{m0}B_{0p} \rightarrow C_{mp;0}.\\
		A_{m0}B_{1p} \rightarrow C_{mp;1}.

For each function, the computing complexity is reduced to

.. math::
		m\times p\times (2\times 1-1).

Therefore, the required amount of memory is reduced by half. However, the computing time becomes two times longer.

The indices to be sliced are selected from the most common index among the tensor network. The module will also try several attempts to find different ways to cut the tensor network for each trial.   

Find the path with minimal computing cost
-------------------------------------------
After slicing the trial contraction paths, the cost function evaluates the contraction paths. The cost function is based on the number of flops and the size of the tensor network defined as

.. math::
		cost = log_{2}(N_{size})+0.1\times log_{10}(N_{flops})

The path with minimal cost will be selected as the optimal path for computing the tensor network.


Compute the tensor network along the optimal path
----------------------------------------------------
The JDOptTN module will use the optimal contraction path to compute the tensor network. 


	

.. Jingdong Optimizer for Tensor Network
.. =======================================
.. .. automodule:: tedq.JD_opt_tn.JD_opt_tn
.. 	:members:
.. 	:undoc-members:

.. Build tree
.. ============
.. .. automodule:: tedq.JD_opt_tn.build_tree
.. 	:members:
.. 	:undoc-members:

.. Gen Trials
.. ============
.. .. automodule:: tedq.JD_opt_tn.gen_trials
.. 	:members:
.. 	:undoc-members: