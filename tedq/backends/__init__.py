r'''
This module contains backends for real executation of quantum circuit
'''

# pylint: disable=invalid-name

from .pytorch_backend import PyTorchBackend
from .jax_backend import JaxBackend
from .hardware_backend import HardwareBackend_quafu, HardwareBackend_qiskit
from .qudio_backend import QUDIOBackend
