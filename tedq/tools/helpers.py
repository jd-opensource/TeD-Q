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


r'''
Helper functions
'''

import torch

def dec_to_bin(x, size):
    r'''
    converting a decimal number into binary format.

    **Example**

    .. code-block:: python3
        x = 3
        size = 2
        >>> dec_to_bin(3, 2)
        [1, 1]
    '''
    # pylint: disable=invalid-name
    import math
    size = int(math.log2(size))
    n = bin(x)[2:]
    n = n.zfill(size)
    result = [int(ix) for ix in n]
    return result


def rescale_state(state):
    r'''
    rescale state
    '''
    probs_tensor = torch.abs(state) ** 2
    scale = torch.sqrt(torch.sum(probs_tensor))
    scale_state = [s/scale for s in state]

    return scale_state