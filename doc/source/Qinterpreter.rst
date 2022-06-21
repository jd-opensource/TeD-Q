==============================
Qinterpreter
==============================
---------
Overview
---------
The :class:`OperatorBase` class is a base class inherited by both the :class:`ObservableBase` class and the
:class:`GateBase` class. Instantiated subclasses of these two classes are building blocks to implement the quantum circuit in the TeD-Q package.

* Each :class:`~.GateBase` subclass represents a quantum gate. Gate parameter(s) (like theta and phi) and the qubit(s) this gate acts on are needed to instantiate it.
* Each  :class:`~.ObservableBase` subclass represents an observable that needs to be measured. Observable parameter(s) (like theta and phi) and the measurement qubit(s) are required to form a further measurement process.

-----------------------
Gates and Measurements
-----------------------

Abstract Base Class
====================

.. automodule:: tedq.QInterpreter.operators.ops_abc
 	:members:

Qauntum Gates
====================

.. automodule:: tedq.QInterpreter.operators.qubit
	:members:

Quantum Mearsurements
========================

.. automodule:: tedq.QInterpreter.operators.measurement
	:members:
	:undoc-members:


-----------------------
Circuits
-----------------------

Storage Base Class
======================

.. automodule:: tedq.QInterpreter.circuits.storage_base
	:members:
	:undoc-members:


Circuit Parser
======================

.. automodule:: tedq.QInterpreter.circuits.Circuit
	:members:
	:undoc-members:

-----------------------
Circuits Viewer
-----------------------

.. automodule:: tedq.QInterpreter.visualization.visualization
	:members:
	:undoc-members: