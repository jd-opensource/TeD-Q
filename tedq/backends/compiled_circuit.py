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
The ``CompiledCircuit`` is the parent class of the JAX and PyTorch backend, e.g.
 ``pytorch_backend`` and ``jax_backend``. The common function are implemented in
  this class and handling of the tensor dot operation are implemented in the
   corresponding child class. The main function is to get the result of the circuit
    based on the specific output type with specific computation method.
"""

# pylint: disable=line-too-long, trailing-whitespace, too-many-lines, too-many-instance-attributes, too-few-public-methods, pointless-string-statement

from collections import OrderedDict
import math

from tedq.tensor_network import gen_tensor_networks
from tedq.quantum_error import QuantumValueError


# pylint: disable=too-many-public-methods
class CompiledCircuit:
    r'''
    Base class for backend, this class transfer Circuit to a executable one with information of backend, differential method, interface specified.

    Args:
        backend (string): Name of the computation backend -- ``jax`` or ``pytorch``
        circuit (.Circuit): Circuit to be computed.
        use_cotengra (Bool): Whether to use cotengra optimizer or not.
        use_jdopttn (Bool): Whether to use cotengra optimizer or not.
        hyper_opt (dict): TODO slice options

    '''

    # pylint: disable=too-many-arguments, too-many-locals, too-many-branches, too-many-statements
    def __init__(self, backend, circuit, use_cotengra=False, use_jdopttn=False, tn_mode=False, hyper_opt=None, tn_simplify=False):

        self._use_cotengra = use_cotengra
        self._use_jdopttn = use_jdopttn

        outer_path_finder = 0
        if use_cotengra and use_jdopttn:
            outer_path_finder = 1
            raise ValueError("Error!!!! can not use contengra, opt_einsum and cyc at the same time!")

        self._tn_mode = tn_mode
        if outer_path_finder and tn_mode:
            raise ValueError("Error!!!! can not use contengra, opt_einsum and cyc at the same time!")

        self._tn_simplify = tn_simplify

        self._num_qubits = circuit.num_qubits
        self._backend = backend
        self._operatorname = OrderedDict()
        self._appliedqubits = OrderedDict()
        self._operatorparams = OrderedDict()
        self._trainableparams = OrderedDict()
        self._operators = OrderedDict()
        self._matrixs = OrderedDict()
        self._operands = []
        self._adjointoperands = []

        for operator in circuit.operators:
            # print(operator)
            self._operators[operator.instance_id] = operator
            self._operatorname[operator.instance_id] = operator.name
            self._appliedqubits[operator.instance_id] = operator.qubits
            self._operatorparams[operator.instance_id] = operator.parameters
            self._trainableparams[operator.instance_id] = operator.trainable_params
            self._matrixs[operator.instance_id] = operator.matrix
        self._measurements = circuit.measurements

        self._gate_tensors = {
            "I": self.get_I_tensor,
            "Hadamard": self.get_Hadamard_tensor,
            "PauliX": self.get_PauliX_tensor,
            "PauliY": self.get_PauliY_tensor,
            "PauliZ": self.get_PauliZ_tensor,
            "S": self.get_S_tensor,
            "T": self.get_T_tensor,
            "SX": self.get_SX_tensor,
            "CNOT": self.get_CNOT_tensor,
            "CZ": self.get_CZ_tensor,
            "CY": self.get_CY_tensor,
            "SWAP": self.get_SWAP_tensor,
            "CSWAP": self.get_CSWAP_tensor,
            "Toffoli": self.get_Toffoli_tensor,
            "RX": self.get_RX_tensor,
            "RY": self.get_RY_tensor,
            "RZ": self.get_RZ_tensor,
            "Rot": self.get_Rot_tensor,
            "PhaseShift": self.get_PhaseShift_tensor,
            "ControlledPhaseShift": self.get_ControlledPhaseShift_tensor,
            "CRX": self.get_CRX_tensor,
            "CRY": self.get_CRY_tensor,
            "CRZ": self.get_CRZ_tensor,
        }

        vector_lenght = 2 ** self._num_qubits
        flops = 0. # flops of this quantum circuit
        # A: m*p  B: n*p; flops of A*B: m*n*(2*p-1)


        self._axeslist = []
        self._permutationlist = []
        for _, qbts in self._appliedqubits.items():  # idx qubits

            len_qbts = len(qbts)
            perms = []
            gate_pos = []

            if len_qbts == 1:
                gate_pos = [1]
                perms = (
                    list(range(1, qbts[0] + 1))
                    + [0]
                    + list(range(qbts[0] + 1, self._num_qubits))
                )

                flops += vector_lenght*(2*2-1)


            elif len_qbts == 2:
                gate_pos = [2, 3]
                res_right = list(range(self._num_qubits))
                res_right = [x for x in res_right if x not in qbts]
                # print(res_right)
                # print(qbts)
                # res_right.pop(qbts[0])#cannot use pop! since after pop, the elements changed
                # print(res_right)
                # res_right.pop(qbts[1])
                # print(res_right)
                dot_result = qbts + res_right
                # print(dot_result)
                for i in range(self._num_qubits):
                    # print(i)
                    perms.append(dot_result.index(i))

                flops += vector_lenght*(2*4-1)


            elif len_qbts == 3:
                gate_pos = [3, 4, 5]
                res_right = list(range(self._num_qubits))
                res_right = [x for x in res_right if x not in qbts]
                dot_result = qbts + res_right
                for i in range(self._num_qubits):
                    perms.append(dot_result.index(i))

                flops += vector_lenght*(2*8-1)


            else:
                raise QuantumValueError("Error on __init__ of Compiled_Circuit, unknown gate with num_qubits larger than 3")
            
            state_pos = qbts
            self._axeslist.append((gate_pos, state_pos))
            self._permutationlist.append(perms)


        self._axeslist.reverse()
        self._permutationlist.reverse()

        print("log10(flops) of this quantum circuit:  ", math.log10(flops+1.0e-10))

        #print(self._tn_mode)
        if self._use_cotengra or self._use_jdopttn or self._tn_mode:
            #print(self._tn_mode)
            '''
            self._cotengra_operands = []  # each measurement will have its own operand
            _layer_ids = list(range(self._num_qubits))
            _current_ids = self._num_qubits - 1
            _input_indices = []
            for i in _layer_ids:
                _input_indices.append(list(get_symbol(i)))

            for _, qbts in self._appliedqubits.items():  # idx qubits

                len_qbts = len(qbts)

                if len_qbts == 1:
                    _current_ids = _current_ids + 1
                    _input_indices.append(
                        [get_symbol(_layer_ids[qbts[0]]), get_symbol(_current_ids)])
                    _layer_ids[qbts[0]] = _current_ids

                elif len_qbts == 2:
                    _current_ids = _current_ids + 2
                    _input_indices.append([get_symbol(_layer_ids[qbts[0]]), get_symbol(
                        _layer_ids[qbts[1]]), get_symbol(_current_ids - 1), get_symbol(_current_ids)])
                    _layer_ids[qbts[0]] = _current_ids - 1
                    _layer_ids[qbts[1]] = _current_ids

                elif len_qbts == 3:
                    _current_ids = _current_ids + 3
                    _input_indices.append([get_symbol(_layer_ids[qbts[0]]), get_symbol(_layer_ids[qbts[1]]), get_symbol(
                        _layer_ids[qbts[2]]), get_symbol(_current_ids - 2), get_symbol(_current_ids - 1), get_symbol(_current_ids)])
                    _layer_ids[qbts[0]] = _current_ids - 2
                    _layer_ids[qbts[1]] = _current_ids - 1
                    _layer_ids[qbts[2]] = _current_ids

            for measurement in self._measurements:

                _output_indices_i = []

                _current_ids_i = deepcopy(_current_ids)
                _input_indices_i = deepcopy(_input_indices)
                _layer_ids_i = deepcopy(_layer_ids)

                if measurement.return_type is State:
                    _output_indices_i = list(get_symbol(
                        _layer_ids_i[i]) for i in range(self._num_qubits))

                else:

                    if measurement.return_type is Expectation:
                        _current_ids_i = _current_ids_i + 1
                        layer = measurement.obs.qubits[0]
                        _input_indices_i.append(
                            [get_symbol(_layer_ids_i[layer]), get_symbol(_current_ids_i)])
                        _layer_ids_i[layer] = _current_ids_i

                    if measurement.return_type is Probability:
                        if measurement.qubits is None:
                            pass
                        else:
                            _output_indices_i = list(get_symbol(
                                _layer_ids_i[qbts]) for qbts in measurement.qubits)

                    for _, qbts in reversed(self._appliedqubits.items()):  # idx qubits

                        len_qbts = len(qbts)

                        if len_qbts == 1:
                            _current_ids_i = _current_ids_i + 1
                            _input_indices_i.append(
                                [get_symbol(_layer_ids_i[qbts[0]]), get_symbol(_current_ids_i)])
                            _layer_ids_i[qbts[0]] = _current_ids_i

                        elif len_qbts == 2:
                            _current_ids_i = _current_ids_i + 2
                            _input_indices_i.append([get_symbol(_layer_ids_i[qbts[0]]), get_symbol(
                                _layer_ids_i[qbts[1]]), get_symbol(_current_ids_i - 1), get_symbol(_current_ids_i)])
                            _layer_ids_i[qbts[0]] = _current_ids_i - 1
                            _layer_ids_i[qbts[1]] = _current_ids_i

                        elif len_qbts == 3:
                            _current_ids_i = _current_ids_i + 3
                            _input_indices_i.append([get_symbol(_layer_ids_i[qbts[0]]), get_symbol(_layer_ids_i[qbts[1]]), get_symbol(
                                _layer_ids_i[qbts[2]]), get_symbol(_current_ids_i - 2), get_symbol(_current_ids_i - 1), get_symbol(_current_ids_i)])
                            _layer_ids_i[qbts[0]] = _current_ids_i - 2
                            _layer_ids_i[qbts[1]] = _current_ids_i - 1
                            _layer_ids_i[qbts[2]] = _current_ids_i

                for i in _layer_ids_i:
                    _input_indices_i.append(list(get_symbol(i)))

                _size_dict_i = dict()

                for ids in _input_indices_i:
                    for symbol in ids:
                        _size_dict_i[symbol] = 2
                for symbol in _output_indices_i:
                    _size_dict_i[symbol] = 2

                self._cotengra_operands.append(
                    (_input_indices_i, _output_indices_i, _size_dict_i)
                    )
            '''

            if self._tn_simplify == -1:
                self._appliedqubits[314159] = [28, 7, 4, 13, 50, 0, 14, 2, 7, 4, 13, 6]

            self._tensor_networks = gen_tensor_networks(self._num_qubits, self._operators, self._appliedqubits, self._measurements)

            self._cotengra_operands = []

            for tn in self._tensor_networks:  # pylint: disable=invalid-name

                if self._tn_simplify:
                    if self._backend == 'pytorch':
                        tn.simplify()
                    else:
                        raise ValueError("JAX backend does not support tensor network simplify yet!!!")

                input_indices = tn.input_indices
                output_indices = tn.output_indices
                size_dict = tn.size_dict

                self._cotengra_operands.append(
                    (input_indices, output_indices, size_dict)
                    )
            


            if self._use_cotengra:
                #print(hyper_opt)
                if not isinstance(hyper_opt, dict):
                    if hyper_opt is not None:
                        print("warning!, input hyper_opt should be a dict type")
                    hyper_opt = {}

                methods = hyper_opt.get('methods', ['kahypar'])
                max_repeats = hyper_opt.get('max_repeats', 128)
                progbar = hyper_opt.get('progbar', False)
                minimize = hyper_opt.get('minimize', 'flops')
                score_compression = hyper_opt.get('score_compression', 0.5)
                slicing_opts = hyper_opt.get('slicing_opts', None)

                #print(progbar)
                #import cotengra as ctg  # pylint: disable=import-outside-toplevel
                ctg = self._use_cotengra
                self._optimize_order_trees = []
                for (input_indices, output_indices, size_dict) in self._cotengra_operands:
                    opt = ctg.HyperOptimizer(
                        methods=methods,
                        max_repeats=max_repeats,
                        progbar=progbar,
                        minimize=minimize,
                        score_compression=score_compression,  # deliberately make the optimizer try many methods
                        slicing_opts = slicing_opts
                    )
                    tree = opt.search(input_indices, output_indices, size_dict)
                    self._optimize_order_trees.append(tree)

            elif self._use_jdopttn:
                #print(hyper_opt)
                if not isinstance(hyper_opt, dict):
                    if hyper_opt is not None:
                        print("warning!, input hyper_opt should be a dict type")
                    hyper_opt = {}

                max_repeats = hyper_opt.get('max_repeats', 128)
                search_parallel = hyper_opt.get('search_parallel', True)
                slicing_opts = hyper_opt.get('slicing_opts', None)

                #print("compiled_circuit's search_parallel:   ", search_parallel)

                #from tedq.JD_opt_tn import JDOptTN  # pylint: disable=import-outside-toplevel
                JDOptTN = self._use_jdopttn
                self._optimize_order_trees = []
                for (input_indices, output_indices, size_dict) in self._cotengra_operands:
                    tree = JDOptTN(
                        input_indices, size_dict, output=output_indices, imbalance = 0.2, max_repeats = max_repeats, search_parallel = search_parallel, slicing_opts = slicing_opts)
                    self._optimize_order_trees.append(tree)

            else:
                # using default opt_einsum method
                #print("here")
                from tedq.oe_wrapper import OEWrapper
                self._optimize_order_trees = []
                for (input_indices, output_indices, size_dict) in self._cotengra_operands:
                    tree = OEWrapper(
                        input_indices, size_dict, output=output_indices)
                    #print(tree.contract_expression)
                    self._optimize_order_trees.append(tree)                


    def execute(self, *params):
        '''
        execute quantum circuit to get the measurement result.
        '''

        self._update_parameters(*params)

    def _parser_circuit(self):
        '''
        Get tensor, name and parameters of each gate from the `Circuit` for ``Wave function vector method``.
        '''
        self._operands = []
        for idx, trnblepars in self._trainableparams.items():

            len_tp = len(trnblepars)

            if len_tp > 0:
                name = self._operatorname[idx]
                parameters = self._operatorparams[idx]
                ts = self._tensor_of_gate(name, parameters)

            else:
                matrix = self._matrixs[idx]
                num_qubits = len(self._appliedqubits[idx])
                ts = self._matrix_to_tensor(matrix, num_qubits)

            self._operands.append(ts)


        self._operands.reverse()

    def _parse_circuit_cotengra(self):
        '''
        Get tensor, name and parameters of each gate from the `Circuit` for ``Tensor network method``. The series of operands and the series of their complex-conjugated operands are stored seperately for different types of circuit output.
        '''
        #print("_parse_circuit_cotengra:  ", self._device)
        self._operands = []
        self._adjointoperands = []

        for idx, trnblepars in self._trainableparams.items():

            len_tp = len(trnblepars)

            if len_tp > 0:
                name = self._operatorname[idx]
                parameters = self._operatorparams[idx]
                ts = self._tensor_of_gate(name, parameters)

            else:
                matrix = self._matrixs[idx]
                num_qubits = len(self._appliedqubits[idx])
                ts = self._matrix_to_tensor(matrix, num_qubits)

            self._operands.append(ts)
            self._adjointoperands.append(self.complex_conjugate(ts))

        self._adjointoperands.reverse()

    def _tensor_of_gate(self, gatename, *params):
        '''
        Get the corresponding tensor of gate with the specific parameters.

        Args:
            params (array): Array of the parameters.

        '''
        # for par in params[0]:
        # print(par.requires_grad)
        # print(self._gate_tensors[gatename](*params).requires_grad)
        #print("_tensor_of_gate:  ", self._device)
        ts = self._gate_tensors[gatename](*params)
        return ts

    def _matrix_to_tensor(self, matrix, num_qubits):
        '''
        converting a numpy matrix to corresponding backend tensor
        '''

        raise NotImplementedError


    def _update_parameters(self, *params):
        '''
        Update the parameters of trainable gate according to new input parameters. The new input parameters are passed to this function when the `compiledCircuit` is called.

        Args:
            params (array): Array of the new parameters.

        **Example**

        .. code-block:: python3

            def circuitDef(*params):
                qai.RX(params[0], qubits=[1])
                qai.Hadamard(qubits=[0])
                qai.CNOT(qubits=[0,1])
                qai.RY(params[1], qubits=[0])
                return qai.expval(qai.PauliZ(qubits=[0]))
            #compile the quantum circuit
            circuit = qai.Circuit(circuitDef, 2, 0.54, 0.12)
            my_compilecircuit = circuit.compilecircuit(backend="jax")
    
        Compiled circuit will called this function and evaluate the circuit based on the updated parameters:

        >>> my_compilecircuit(0.54, 0.12)
        [DeviceArray(-9.4627275e-09, dtype=float32)]

        '''

        # print("_update_parameters: ", params, len(params))
        count = 0
        # print(self._operatorparams, len(self._operatorparams))

        for idx, trnblepars in self._trainableparams.items():
            len_tp = len(trnblepars)
            # print(len_tp)
            # print(count)
            # print(len(params))
            # print(params)
            # print(idx, self._operatorparams[idx],trnblepars)

            if len_tp > 0:
                #for i in range(len_tp):# trnblepars is list of indices of trainable parameters of the gate
                for i in range(len_tp):
                    pos = trnblepars[i]
                    # print(self._operatorparams[idx])
                    # print(self._operatorparams[idx][i])
                    # print(params[count+i].requires_grad)

                    # print(idx, pos, count, i, len(params), len(self._operatorparams[idx]))

                    self._operatorparams[idx][pos] = params[count + i]
                count = count + len_tp
        if len(params) != count:
            raise ValueError(f'Error!!!! number of parameters are not matched!! required {count} but {len(params)} are given')

    @property
    def operators(self):
        '''
        Obtain operators of circuit
        '''
        ops = []
        for _, opt in self._operators.items():
            ops.append(opt)
        return ops

    @property
    def gates_names(self):
        '''
        Obtain name of gates of circuit
        '''
        gatesnames = []
        for _, name in self._operatorname.items():
            gatesnames.append(name)
        return gatesnames

    @property
    def gate_parameters(self):
        '''
        Obtain parameters of circuit
        '''
        gatesparams = []
        for _, pars in self._operatorparams.items():
            gatesparams.append(pars)
        return gatesparams

    @property
    def parameters(self):# be careful, this function only can be called before executing __call__ function for jax backend because of jit, this is for CY's drawing, need to be improved
        '''
        All parameters, including trainable parameters and parameters that do not need to be trained
        Obtain parameters of circuit
        '''
        parameters = []
        for _, pars in self._operatorparams.items():
            parameters.extend(pars)
        return parameters

    @property
    def trainable_parameters(self):# 
        '''
        Trainable parameters
        Obtain parameters of circuit
        '''
        trainable_parameters = []
        for idx, pars in self._trainableparams.items():
            len_pars = len(pars)
            if len_pars:
                for i in pars:
                    trainable_parameters.append(self._operatorparams[idx][i])
        return trainable_parameters

    @property
    def qubits(self):
        '''
        Obtain qubits of circuit
        '''
        appliedqubits = []
        for _, qbts in self._appliedqubits.items():  # idx qubits
            appliedqubits.append(qbts)
        return appliedqubits

    @property
    def interface(self):
        '''
        return the interface
        '''
        return self._interface  # pylint: disable=no-member

    @property
    def diff_method(self):
        '''
        return the differential method
        '''
        return self._diff_method  # pylint: disable=no-member

    @property
    def backend(self):
        '''
        return the computation backend
        '''
        return self._backend

    @property
    def measurements(self):
        """
        return the measurements
        """
        return self._measurements

    def complex_conjugate(self, ts):
        '''
        Get complex conjugate of that tensor
        '''

        raise NotImplementedError

    # pylint: disable=invalid-name

    @classmethod
    def get_I_tensor(cls, paramslist):
        '''
        Get corresponding tensor
        '''

        raise NotImplementedError

    @classmethod
    def get_Hadamard_tensor(cls, paramslist):
        '''
        Get corresponding tensor
        '''

        raise NotImplementedError

    @classmethod
    def get_PauliX_tensor(cls, paramslist):
        '''
        Get corresponding tensor
        '''

        raise NotImplementedError

    @classmethod
    def get_PauliY_tensor(cls, paramslist):
        '''
        Get corresponding tensor
        '''

        raise NotImplementedError

    @classmethod
    def get_PauliZ_tensor(cls, paramslist):
        '''
        Get corresponding tensor
        '''

        raise NotImplementedError

    @classmethod
    def get_S_tensor(cls, paramslist):
        '''
        Get corresponding tensor
        '''

        raise NotImplementedError

    @classmethod
    def get_T_tensor(cls, paramslist):
        '''
        Get corresponding tensor
        '''

        raise NotImplementedError

    @classmethod
    def get_SX_tensor(cls, paramslist):
        '''
        Get corresponding tensor
        '''

        raise NotImplementedError

    @classmethod
    def get_CNOT_tensor(cls, paramslist):
        '''
        Get corresponding tensor
        '''

        raise NotImplementedError

    @classmethod
    def get_CZ_tensor(cls, paramslist):
        '''
        Get corresponding tensor
        '''

        raise NotImplementedError

    @classmethod
    def get_CY_tensor(cls, paramslist):
        '''
        Get corresponding tensor
        '''

        raise NotImplementedError

    @classmethod
    def get_SWAP_tensor(cls, paramslist):
        '''
        Get corresponding tensor
        '''

        raise NotImplementedError

    @classmethod
    def get_CSWAP_tensor(cls, paramslist):
        '''
        Get corresponding tensor
        '''

        raise NotImplementedError

    @classmethod
    def get_Toffoli_tensor(cls, paramslist):
        '''
        Get corresponding tensor
        '''

        raise NotImplementedError

    @classmethod
    def get_RX_tensor(cls, paramslist):
        '''
        Get corresponding tensor
        '''

        raise NotImplementedError

    @classmethod
    def get_RY_tensor(cls, paramslist):
        '''
        Get corresponding tensor
        '''

        raise NotImplementedError

    @classmethod
    def get_RZ_tensor(cls, paramslist):
        '''
        Get corresponding tensor
        '''

        raise NotImplementedError

    @classmethod
    def get_Rot_tensor(cls, paramslist):
        '''
        Get corresponding tensor
        '''

        raise NotImplementedError

    @classmethod
    def get_PhaseShift_tensor(cls, paramslist):
        '''
        Get corresponding tensor
        '''

        raise NotImplementedError

    @classmethod
    def get_ControlledPhaseShift_tensor(cls, paramslist):
        '''
        Get ControlledPhaseShift tensor
        '''

        raise NotImplementedError

    @classmethod
    def get_CRX_tensor(cls, paramslist):
        '''
        Get CRX tensor
        '''

        raise NotImplementedError

    @classmethod
    def get_CRY_tensor(cls, paramslist):
        '''
        Get CRY tensor
        '''

        raise NotImplementedError

    @classmethod
    def get_CRZ_tensor(cls, paramslist):
        '''
        Get corresponding tensor
        '''

        raise NotImplementedError

'''
def get_symbol(i):
    """Get the symbol corresponding to int ``i`` - runs through the usual 52
    letters before resorting to unicode characters, starting at ``chr(192)``.   

    Examples:
    
    >>> get_symbol(1)
    'b' 

    >>> get_symbol(200)
    'Ŕ' 

    >>> get_symbol(20000)
    '京'
    """
    _einsum_symbols_base = 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ'
    if i < 52:
        return _einsum_symbols_base[i]
    return chr(i + 140)
'''