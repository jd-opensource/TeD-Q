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
Test file for ray parallel.
"""
import numpy as np
import pytest
from tedq.ray_parallel.parallel import get_ray



class Test_RayParallel():
    r'''
    '''

    def test_ray_initialization(self):
        r'''
        '''
        input_num_cpu = 2
        ray = get_ray(input_num_cpu)
        assert ray.is_initialized()

        num_cpu = int(ray.available_resources()['CPU'])
        assert input_num_cpu == num_cpu
        #ray.shutdown()

        print("Test ray initialization ok!")


    def test_ray_shutdown(self):
        r'''
        '''
        input_num_cpu = 2
        ray = get_ray(input_num_cpu)
        #assert ray.is_initialized()

        ray.shutdown()
        assert ray.is_initialized() == False

        print("Test ray shutdown ok!")
