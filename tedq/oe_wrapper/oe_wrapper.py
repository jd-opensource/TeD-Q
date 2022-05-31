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
opt_einsum optimizer for finding best tensor network contraction path.
"""

# pylint: disable=line-too-long, trailing-whitespace, too-many-lines, too-many-instance-attributes, too-few-public-methods, pointless-string-statement

import opt_einsum as oe

class OEWrapper():
    r'''
    opt_einsum wrapper.
    '''
    def __init__(self, input_indices, size_dict, output=None):
        r'''
        opt_einsum wrapper.
        '''
        if output is None:
            output = []    

        sizes = []
        for indices in input_indices:
            size = []
            for index in indices:
                size.append(size_dict[index])
            size = tuple(size)
            sizes.append(size)    

        sizes = tuple(sizes)    

        einsum_str_in = ''
        for indices in input_indices:
            einsum_str_in += ''.join(indices)
            einsum_str_in += ','    

        einsum_str_in = einsum_str_in[:-1]
        einsum_str_in += '->'    

        einsum_str_out = ''.join(output)    

        einsum_str = einsum_str_in + einsum_str_out

        self.contract_expression = oe.contract_expression(einsum_str, *sizes)

    def contract(self, arrays):
        r'''
        Do the contraction.
        '''
        arrays = tuple(arrays)
        result = self.contract_expression(*arrays)
        return result
