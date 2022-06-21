------------
Interface
------------
To provide the support for back-propagation with PyTorch and JAX, the JAX computation are wrapped into a simple customized `forward` and `backward` function of PyTorch. In this case, JAX module are treated as a custom-defined function in PyTorch, which provide good compatibility to PyTorch and high computation speed by JAX.

PyTorch interface
==================
.. automodule:: tedq.interface.pytorch_interface
	:members:


