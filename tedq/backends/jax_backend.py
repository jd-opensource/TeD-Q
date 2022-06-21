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

r'''
This is the backend embeded the JAX computation module into tedq.
 The backend is used for the computation of simulation of quantum circuit,
 as well as finding the gradient of output result among the input parameters.
 To make the simulation of quantum circuit compatible to the PyTorch machine
 learning module, the tedq also support to include a PyTorch interface to JAX
 backend. Detailed can be found in ``TODO``

There're three supported methods to calculate the gradient
    * Parameters-shift method (TBD)
    * Finite-differential (TBD)
    * Back-propagation method

'''

# pylint: disable=line-too-long, trailing-whitespace, too-many-lines, too-many-instance-attributes, too-few-public-methods, too-many-arguments
# pylint: disable=no-member, no-name-in-module, too-many-public-methods, too-many-statements, too-many-branches, too-many-locals

from copy import deepcopy
import jax
from jax import numpy as jnp
from tedq.QInterpreter.operators.measurement import Expectation, Probability, State
from tedq.interface import PytorchInterface
from .compiled_circuit import CompiledCircuit

JCOMPLEX = jnp.complex64


class JaxBackend(CompiledCircuit):
    '''
    This is the backend embeded the JAX computation module into tedq.
    The computation function can be compiled by JIT to increase the computation speed. Besides, PyTorch interface is also included to provide compatibility to Pytorch ML module, which can be turned on by `interface` option.

    Args:
        backend (string): Name of the computation backend -- ``jax`` or ``pytorch``
        circuit (.Circuit): Circuit to be computed.
        use_cotengra (Bool): Whether to use cotengra optimizer or not.
        use_jdopttn (Bool): Whether to use cotengra optimizer or not.
        hyper_opt (dict): TODO slice options
        kwargs (dict): Other keyword arguments
    '''
    #pytorch backend only support list of measurements of the same dimension,
    #since finally a torch.stack function is used to combine all the measurement result into a torch.tensor object

    def __init__(self, backend, circuit, use_cotengra=False, use_jdopttn=False, tn_mode=False, hyper_opt=None, tn_simplify = False, **kwargs):

        if tn_mode:
            print("opt_einsum's contract method will conflict with jax.jit or jax.grad, change to state vector propagation mode!")
            tn_mode = False
        super().__init__(backend, circuit, use_cotengra=use_cotengra,
                         use_jdopttn=use_jdopttn, tn_mode=tn_mode, hyper_opt=hyper_opt, tn_simplify = tn_simplify)

        self._diff_method = kwargs.get("diff_method", "back_prop")
        self._interface = kwargs.get("interface", "jax")

        if self._diff_method != "back_prop":
            raise ValueError(
                f'{self._diff_method}: jax only support built-in back_prop method!'
            )

        if self._interface == "jax":
            self.jit_execute = self.execute  #jax.jit(self.execute)
            #self.jit_execute = self.execute

        elif self._interface == "pytorch":
            self.execute_func = PytorchInterface.apply # forward function of pyTorch

            jit_execute = self.execute  #jax.jit(self.execute)

            def process_execute(input_params):
                flatten_input = list(element.flatten().squeeze()
                                     for element in input_params)
                if jnp.size(flatten_input[0]) > 1:
                    args = tuple(jnp.concatenate(flatten_input, 0), )
                else:
                    args = tuple(flatten_input, )
                return tuple(jit_execute(*args), )

            new_execute = process_execute  #jax.jit(process_execute)
            jaccobian = jax.jacrev(new_execute)  #jax.jit(jax.jacrev(new_execute)) # get jacobian
            self.interface_kwargs = {}
            self.interface_kwargs["execute_func"] = new_execute
            self.interface_kwargs["jaccobian_func"] = jaccobian

        else:
            raise ValueError(
                f'{self._interface}: unknown interface for jax_backend!'
            )

        if not (self._use_cotengra or self._use_jdopttn):
            self.svpm_mode_jit = jax.jit(self.svpm_mode)

    def __call__(self, *params):
        '''
        internal call function
        '''

        if self._interface == "jax":
            call_fcn = self.jit_execute(*params)

        elif self._interface == "pytorch":
            call_fcn = self.execute_func(self.interface_kwargs, *params)

        else:
            raise ValueError(
                f'{self._interface}: unknown interface for jax_backend!'
            )

        return call_fcn

    def execute(self, *params):
        '''
        execution function
        '''

        super().execute(*params)

        if self._use_cotengra:  # pylint: disable=no-else-return
            self._parse_circuit_cotengra() # parse tensor value
            results = []
            for i, _ in enumerate(self.measurements):
                arrays = []
                zero_state = [jnp.array([1., 0.], dtype=JCOMPLEX)
                              for _ in range(self._num_qubits)]
                arrays.extend(zero_state)
                arrays.extend(self._operands)
                if self.measurements[i].return_type is Expectation:
                    arrays.append(self._tensor_of_gate(
                        self.measurements[i].obs.name, []))
                    arrays.extend(self._adjointoperands)
                    arrays.extend(zero_state)
                    result = self._optimize_order_trees[i].contract(
                        arrays, backend='jax')
                    result = jnp.squeeze(jnp.real(result))
                    results.append(result.reshape(()))
                if self.measurements[i].return_type is Probability:
                    arrays.extend(self._adjointoperands)
                    arrays.extend(zero_state)
                    result = self._optimize_order_trees[i].contract(
                        arrays, backend='jax')
                    result = jnp.squeeze(jnp.real(result))
                    results.append(result)
                if self.measurements[i].return_type is State:
                    arrays.extend(zero_state)
                    result = self._optimize_order_trees[i].contract(
                        arrays, backend='jax')
                    results.append(result)
            return results

        elif self._use_jdopttn:
            self._parse_circuit_cotengra()
            results = []
            for i, _ in enumerate(self.measurements):
                arrays = []
                zero_state = [jnp.array([1., 0.], dtype=JCOMPLEX)
                              for _ in range(self._num_qubits)]

                if self._tn_simplify == -1:
                    zero_state = [jnp.array([1. / jnp.sqrt(2.), 1. / jnp.sqrt(2.)], dtype=JCOMPLEX)
                                  for _ in range(self._num_qubits)]

                arrays.extend(zero_state)
                arrays.extend(self._operands)
                if self.measurements[i].return_type is Expectation:
                    arrays.append(self._tensor_of_gate(
                        self.measurements[i].obs.name, []))
                    arrays.extend(self._adjointoperands)
                    arrays.extend(zero_state)
                    result = self._optimize_order_trees[i].contract(
                        arrays, backend='jax')
                    result = jnp.squeeze(jnp.real(result))
                    results.append(result.reshape(()))
                if self.measurements[i].return_type is Probability:
                    arrays.extend(self._adjointoperands)
                    arrays.extend(zero_state)
                    result = self._optimize_order_trees[i].contract(
                        arrays, backend='jax')
                    result = jnp.squeeze(jnp.real(result))
                    results.append(result)
                if self.measurements[i].return_type is State:
                    arrays.extend(zero_state)
                    result = self._optimize_order_trees[i].contract(
                        arrays, backend='jax')
                    results.append(result)
            return results

        elif self._tn_mode:
            self._parse_circuit_cotengra()
            results = []
            for i, _ in enumerate(self.measurements):
                arrays = []
                zero_state = [jnp.array([1., 0.], dtype=JCOMPLEX)
                              for _ in range(self._num_qubits)]
                arrays.extend(zero_state)
                arrays.extend(self._operands)
                if self.measurements[i].return_type is Expectation:
                    arrays.append(self._tensor_of_gate(
                        self.measurements[i].obs.name, []))
                    arrays.extend(self._adjointoperands)
                    arrays.extend(zero_state)
                    result = self._optimize_order_trees[i].contract(
                        arrays)
                    result = jnp.squeeze(jnp.real(result))
                    results.append(result.reshape(()))
                if self.measurements[i].return_type is Probability:
                    arrays.extend(self._adjointoperands)
                    arrays.extend(zero_state)
                    result = self._optimize_order_trees[i].contract(
                        arrays)
                    result = jnp.squeeze(jnp.real(result))
                    results.append(result)
                if self.measurements[i].return_type is State:
                    arrays.extend(zero_state)
                    result = self._optimize_order_trees[i].contract(
                        arrays)
                    results.append(result)
            return results

        else:
            results = self.svpm_mode()  # self.svpm_mode_jit() cannot use, otherwise cannot calculate gradient, need to upgrade
            return results


    def svpm_mode(self):  # state vector propagation mode
        r'''
        state vector propagation mode
        '''
        self._parser_circuit()
        initstate = self.get_initstate()
        self._operands.append(initstate)
        axeslist = deepcopy(self._axeslist)
        permutationlist = deepcopy(self._permutationlist)

        self._operands = core_contract(self._operands, axeslist, permutationlist)

        results = self.get_measurement_results(self._operands[0])
        return results



    def get_measurement_results(self, state):
        '''
        from the final quantum circuit state, calculate the measurement result
        '''
        results = []
        for meas in self.measurements:
            if meas.return_type is Expectation:
                perms = (
                    list(range(1, meas.obs.qubits[0] + 1))
                    + [0]
                    + list(range(meas.obs.qubits[0] + 1, self._num_qubits))
                )
                tmpt = self._tensor_of_gate(meas.obs.name, [])
                tmpt = jnp.tensordot(
                    tmpt, state, axes=([1], meas.obs.qubits)
                )  # order need to change!
                tmpt = jnp.transpose(tmpt, perms)
                axes = list(range(self._num_qubits))
                tmpt = jnp.tensordot(jnp.conjugate(
                    state), tmpt, axes=(axes, axes))
                result = jnp.squeeze(jnp.real(tmpt))
                results.append(result.reshape(()))

            if meas.return_type is Probability:
                if meas.qubits is None:
                    results.append(jnp.abs(state) ** 2)
                else:
                    probs_tensor = jnp.abs(state) ** 2
                    axes = list(range(self._num_qubits))
                    for qbts in meas.qubits:
                        axes.pop(qbts)
                    if len(axes) != 0:
                        result = jnp.sum(probs_tensor, axis=axes)
                    else:
                        result = probs_tensor
                    results.append(result)

            if meas.return_type is State:
                results.append(state)

        return results

    def get_initstate(self):
        if self._init_state:
            matrix = deepcopy(self._init_state.matrix)
            _b = jnp.asarray(matrix, dtype=JCOMPLEX)
            shape = [2 for _ in range(self._num_qubits)]
            shape = tuple(shape)
            return _b.reshape(shape)
        else:
            return self.default_initstate()

    # in state |00...0>
    def default_initstate(self):
        '''
        Get initial state
        '''
        num_elements = 2 ** self._num_qubits
        #_a = jnp.zeros(num_elements, dtype=JCOMPLEX)
        #_b = jax.ops.index_update(_a, jax.ops.index[0], 1)
        #_b = _a.at[0].set(1)
        array = list(0 for _ in range(num_elements))
        array[0] = 1
        _b = jnp.array(array, dtype=JCOMPLEX)
        #print(_a)
        #print(_b)
        shape = [2 for _ in range(self._num_qubits)]
        return _b.reshape(shape)


    def complex_conjugate(self, ts):
        '''
        Get complex conjugate of that tensor
        '''
        shape = ts.shape
        shape_array = jnp.array(shape)
        prod_shape = jnp.prod(shape_array)
        new_size = jnp.sqrt(prod_shape)
        new_size = jnp.array(new_size, int)
        new_shape = (new_size, new_size)
        
        ts = ts.reshape(new_shape)
        ts = ts.conj()
        ts = ts.T
        ts = ts.reshape(shape)
        return ts

    def _matrix_to_tensor(self, matrix, num_qubits):
        '''
        converting a numpy matrix to corresponding backend tensor
        '''

        ts = jnp.asarray(matrix, dtype=JCOMPLEX)
        shape = [2 for _ in range(2*num_qubits)]
        shape = tuple(shape)
        return ts.reshape(shape)

    @classmethod
    def get_I_tensor(cls, paramslist):
        '''
        Get corresponding tensor of I gate

        See :meth:`~tedq.Backends.compiled_circuit.CompiledCircuit.get_I_tensor` in ``CompiledCircuit`` class for more detailed information.
        '''
        if len(paramslist) != 0:
            raise ValueError("I gate does not need any parameter!")

        return jnp.array(
            [
                [1, 0], 
                [0, 1]
            ], 
            dtype=JCOMPLEX
        )

    @classmethod
    def get_Hadamard_tensor(cls, paramslist):
        '''
        Get corresponding tensor of Hadamard gate

        See :meth:`~tedq.Backends.compiled_circuit.CompiledCircuit.get_Hadamard_tensor` in ``CompiledCircuit`` class for more detailed information.
        '''
        if len(paramslist) != 0:
            raise ValueError("Hadamard gate does not need any parameter!")

        return jnp.array(
            [
                [1. / jnp.sqrt(2.), 1. / jnp.sqrt(2.)], 
                [1. / jnp.sqrt(2.), -1. / jnp.sqrt(2.)]
            ], 
            dtype=JCOMPLEX
        )

    @classmethod
    def get_PauliX_tensor(cls, paramslist):
        '''
        Get corresponding tensor of Pauli-X gate

        See :meth:`~tedq.Backends.compiled_circuit.CompiledCircuit.get_PauliX_tensor` in ``CompiledCircuit`` class for more detailed information.
        '''
        if len(paramslist) != 0:
            raise ValueError("PauliX gate does not need any parameter!")

        return jnp.array(
            [
                [0, 1], 
                [1, 0]
            ], 
            dtype=JCOMPLEX
        )

    @classmethod
    def get_PauliY_tensor(cls, paramslist):
        '''
        Get corresponding tensor of Pauli-Y gate

        See :meth:`~tedq.Backends.compiled_circuit.CompiledCircuit.get_PauliY_tensor` in ``CompiledCircuit`` class for more detailed information.
        '''
        if len(paramslist) != 0:
            raise ValueError("PauliY gate does not need any parameter!")

        return jnp.array(
            [
                [0j, -1j], 
                [1j, 0j]
            ], 
            dtype=JCOMPLEX
        )

    @classmethod
    def get_PauliZ_tensor(cls, paramslist):
        '''
        Get corresponding tensor of Pauli-Z gate

        See :meth:`~tedq.Backends.compiled_circuit.CompiledCircuit.get_PauliZ_tensor` in ``CompiledCircuit`` class for more detailed information.
        '''
        if len(paramslist) != 0:
            raise ValueError("PauliZ gate does not need any parameter!")

        return jnp.array(
            [
                [1, 0], 
                [0, -1]
            ], 
            dtype=JCOMPLEX
        )

    @classmethod
    def get_S_tensor(cls, paramslist):
        '''
        Get corresponding tensor of S gate

        See :meth:`~tedq.Backends.compiled_circuit.CompiledCircuit.get_S_tensor` in ``CompiledCircuit`` class for more detailed information.
        '''
        if len(paramslist) != 0:
            raise ValueError("S gate does not need any parameter!")

        return jnp.array(
            [
                [1, 0], 
                [0, 1j]
            ], 
            dtype=JCOMPLEX
        )

    @classmethod
    def get_T_tensor(cls, paramslist):
        '''
        Get corresponding tensor of T gate

        See :meth:`~tedq.Backends.compiled_circuit.CompiledCircuit.get_T_tensor` in ``CompiledCircuit`` class for more detailed information.
        '''
        if len(paramslist) != 0:
            raise ValueError("T gate does not need any parameter!")

        return jnp.array(
            [
                [1, 0], 
                [0, jnp.exp(1j * jnp.pi / 4)]
            ], 
            dtype=JCOMPLEX
        )

    @classmethod
    def get_SX_tensor(cls, paramslist):
        '''
        Get corresponding tensor of SX gate

        See :meth:`~tedq.Backends.compiled_circuit.CompiledCircuit.get_SX_tensor` in ``CompiledCircuit`` class for more detailed information.
        '''
        if len(paramslist) != 0:
            raise ValueError("SX gate does not need any parameter!")

        return jnp.array(
            [
                [0.5 + 0.5j, 0.5 - 0.5j], 
                [0.5 - 0.5j, 0.5 + 0.5j]
            ], 
            dtype=JCOMPLEX
        )

    @classmethod
    def get_CNOT_tensor(cls, paramslist):
        '''
        Get corresponding tensor of CNOT gate

        See :meth:`~tedq.Backends.compiled_circuit.CompiledCircuit.get_CNOT_tensor` in ``CompiledCircuit`` class for more detailed information.
        '''
        if len(paramslist) != 0:
            raise ValueError("CNOT gate does not need any parameter!")

        return jnp.array(
            [
                [1, 0, 0, 0], 
                [0, 1, 0, 0], 
                [0, 0, 0, 1], 
                [0, 0, 1, 0]
            ], 
            dtype=JCOMPLEX
        ).reshape(2, 2, 2, 2)

    @classmethod
    def get_CZ_tensor(cls, paramslist):
        '''
        Get corresponding tensor of CZ gate

        See :meth:`~tedq.Backends.compiled_circuit.CompiledCircuit.get_CZ_tensor` in ``CompiledCircuit`` class for more detailed information.
        '''
        if len(paramslist) != 0:
            raise ValueError("CZ gate does not need any parameter!")

        return jnp.array(
            [
                [1, 0, 0, 0], 
                [0, 1, 0, 0], 
                [0, 0, 1, 0], 
                [0, 0, 0, -1]
            ], 
            dtype=JCOMPLEX
        ).reshape(2, 2, 2, 2)

    @classmethod
    def get_CY_tensor(cls, paramslist):
        '''
        Get corresponding tensor of CY gate

        See :meth:`~tedq.Backends.compiled_circuit.CompiledCircuit.get_CY_tensor` in ``CompiledCircuit`` class for more detailed information.
        '''
        if len(paramslist) != 0:
            raise ValueError("CY gate does not need any parameter!")

        return jnp.array(
            [
                [1, 0, 0, 0],  
                [0, 1, 0, 0], 
                [0, 0, 0, -1j], 
                [0, 0, 1j, 0]
            ], 
            dtype=JCOMPLEX
        ).reshape(2, 2, 2, 2)

    @classmethod
    def get_SWAP_tensor(cls, paramslist):
        '''
        Get corresponding tensor of swap gate

        See :meth:`~tedq.Backends.compiled_circuit.CompiledCircuit.get_SWAP_tensor` in ``CompiledCircuit`` class for more detailed information.
        '''
        if len(paramslist) != 0:
            raise ValueError("SWAP gate does not need any parameter!")

        return jnp.array(
            [
                [1, 0, 0, 0], 
                [0, 0, 1, 0], 
                [0, 1, 0, 0], 
                [0, 0, 0, 1]
            ], 
            dtype=JCOMPLEX
        ).reshape(2, 2, 2, 2)

    @classmethod
    def get_CSWAP_tensor(cls, paramslist):
        '''
        Get corresponding tensor of C-swap gate

        See :meth:`~tedq.Backends.compiled_circuit.CompiledCircuit.get_CSWAP_tensor` in ``CompiledCircuit`` class for more detailed information.
        '''
        if len(paramslist) != 0:
            raise ValueError("CSWAP gate does not need any parameter!")

        return jnp.array(
            [
                [1, 0, 0, 0, 0, 0, 0, 0],
                [0, 1, 0, 0, 0, 0, 0, 0],
                [0, 0, 1, 0, 0, 0, 0, 0],
                [0, 0, 0, 1, 0, 0, 0, 0],
                [0, 0, 0, 0, 1, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 1, 0],
                [0, 0, 0, 0, 0, 1, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 1],
            ],
            dtype=JCOMPLEX,
        ).reshape(2, 2, 2, 2, 2, 2)

    @classmethod
    def get_Toffoli_tensor(cls, paramslist):
        '''
        Get corresponding tensor of Toffoli gate

        See :meth:`~tedq.Backends.compiled_circuit.CompiledCircuit.get_Toffoli_tensor` in ``CompiledCircuit`` class for more detailed information.
        '''
        if len(paramslist) != 0:
            raise ValueError("Toffoli gate does not need any parameter!")

        return jnp.array(
            [
                [1, 0, 0, 0, 0, 0, 0, 0],
                [0, 1, 0, 0, 0, 0, 0, 0],
                [0, 0, 1, 0, 0, 0, 0, 0],
                [0, 0, 0, 1, 0, 0, 0, 0],
                [0, 0, 0, 0, 1, 0, 0, 0],
                [0, 0, 0, 0, 0, 1, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 1],
                [0, 0, 0, 0, 0, 0, 1, 0],
            ],
            dtype=JCOMPLEX,
        ).reshape(2, 2, 2, 2, 2, 2)

    @classmethod
    def get_RX_tensor(cls, paramslist):
        '''
        Get corresponding tensor of RX gate

        See :meth:`~tedq.Backends.compiled_circuit.CompiledCircuit.get_RX_tensor` in ``CompiledCircuit`` class for more detailed information.
        '''
        theta = paramslist[0]
        _c = jnp.cos(theta / 2)
        _js = 1j * jnp.sin(-theta / 2)
        #print(jnp.array([[_c, _js], [_js, _c]], dtype=JCOMPLEX))
        return jnp.array(
            [
                [_c, _js], 
                [_js, _c]
            ], 
            dtype=JCOMPLEX
        )

    @classmethod
    def get_RY_tensor(cls, paramslist):
        '''
        Get corresponding tensor of RY gate

        See :meth:`~tedq.Backends.compiled_circuit.CompiledCircuit.get_RY_tensor` in ``CompiledCircuit`` class for more detailed information.
        '''
        theta = paramslist[0]
        _c = jnp.cos(theta / 2)
        _s = jnp.sin(theta / 2)
        #print(jnp.array([[_c, -_s], [_s, _c]], dtype=JCOMPLEX))
        return jnp.array(
            [
                [_c, -_s], 
                [_s, _c]
            ], 
            dtype=JCOMPLEX
        )

    @classmethod
    def get_RZ_tensor(cls, paramslist):
        '''
        Get corresponding tensor of RZ gate

        See :meth:`~tedq.Backends.compiled_circuit.CompiledCircuit.get_RZ_tensor` in ``CompiledCircuit`` class for more detailed information.
        '''
        theta = paramslist[0]
        _p = jnp.exp(-0.5j * theta)
        return jnp.array(
            [
                [_p, 0], 
                [0, _p.conjugate()]
            ], 
            dtype=JCOMPLEX
        )

    @classmethod
    def get_Rot_tensor(cls, paramslist):
        '''
        Get corresponding tensor of Rot gate

        See :meth:`~tedq.Backends.compiled_circuit.CompiledCircuit.get_Rot_tensor` in ``CompiledCircuit`` class for more detailed information.
        '''
        theta = paramslist[0]
        phi = paramslist[1]
        omega =paramslist[2]
        _c = jnp.cos(phi / 2.)
        _s = jnp.sin(phi / 2.)
        return jnp.array(
            [
                [jnp.exp(-0.5j * (theta + omega)) * _c, -jnp.exp(0.5j * (theta - omega)) * _s], 
                [jnp.exp(-0.5j * (theta - omega)) * _s, jnp.exp(0.5j * (theta + omega)) * _c]
            ], 
            dtype=JCOMPLEX
        )

    @classmethod
    def get_PhaseShift_tensor(cls, paramslist):
        '''
        Get corresponding tensor of phase-shift gate

        See :meth:`~tedq.Backends.compiled_circuit.CompiledCircuit.get_PhaseShift_tensor` in ``CompiledCircuit`` class for more detailed information.
        '''
        phi = paramslist[0]
        return jnp.array(
            [
                [1, 0], 
                [0, jnp.exp(1j * phi)]
            ], 
            dtype=JCOMPLEX
        )

    @classmethod
    def get_ControlledPhaseShift_tensor(cls, paramslist):
        '''
        Get corresponding tensor of controlled-phase-shift gate

        See :meth:`~tedq.Backends.compiled_circuit.CompiledCircuit.get_ControlledPhaseShift_tensor` in ``CompiledCircuit`` class for more detailed information.
        '''
        phi = paramslist[0]
        return jnp.array(
            [
                [1, 0, 0, 0], 
                [0, 1, 0, 0], 
                [0, 0, 1, 0],
                [0, 0, 0, jnp.exp(1j * phi)]
            ],
            dtype=JCOMPLEX,
        ).reshape(2, 2, 2, 2)

    @classmethod
    def get_CRX_tensor(cls, paramslist):
        '''
        Get corresponding tensor of CRX gate

        See :meth:`~tedq.Backends.compiled_circuit.CompiledCircuit.get_CRX_tensor` in ``CompiledCircuit`` class for more detailed information.
        '''
        theta = paramslist[0]
        _c = jnp.cos(theta / 2)
        _js = 1j * jnp.sin(-theta / 2)
        return jnp.array(
            [
                [1, 0, 0, 0], 
                [0, 1, 0, 0], 
                [0, 0, _c, _js], 
                [0, 0, _js, _c]
            ], 
            dtype=JCOMPLEX
        ).reshape(2, 2, 2, 2)

    @classmethod
    def get_CRY_tensor(cls, paramslist):
        '''
        Get corresponding tensor of CRY gate

        See :meth:`~tedq.Backends.compiled_circuit.CompiledCircuit.get_CRY_tensor` in ``CompiledCircuit`` class for more detailed information.
        '''
        theta = paramslist[0]
        _c = jnp.cos(theta / 2)
        _s = jnp.sin(theta / 2)
        return jnp.array(
            [
                [1, 0, 0, 0], 
                [0, 1, 0, 0], 
                [0, 0, _c, -_s], 
                [0, 0, _s, _c]
            ], 
            dtype=JCOMPLEX
        ).reshape(2, 2, 2, 2)

    @classmethod
    def get_CRZ_tensor(cls, paramslist):
        '''
        Get corresponding tensor of CRZ gate

        See :meth:`~tedq.Backends.compiled_circuit.CompiledCircuit.get_CRZ_tensor` in ``CompiledCircuit`` class for more detailed information.
        '''
        theta = paramslist[0]
        return jnp.array(
            [
                [1, 0, 0, 0],
                [0, 1, 0, 0],
                [0, 0, jnp.exp(-0.5j * theta), 0],
                [0, 0, 0, jnp.exp(0.5j * theta)],
            ],
            dtype=JCOMPLEX,
        ).reshape(2, 2, 2, 2)

def core_contract(operands, axeslist, permutationlist):
    r'''
    core contract for state vector propagation mode
    '''
    _operands = operands
    for _ in range(len(_operands) - 1):
        # print("deal with ?th gate: ", i)
        statevector = _operands.pop(-1)
        appliedgate = _operands.pop(-1)
        axes = axeslist.pop(-1)
        perms = permutationlist.pop(-1)

        # print(statevector.shape)

        # print(axes)
        # print(appliedgate)

        newstate = jnp.tensordot(appliedgate, statevector, axes)

        # print(newstate.shape)
        newstate = jnp.transpose(newstate, perms)
        _operands.append(newstate)
    return _operands