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
This module contains basic function for getting ray and retrieving result from ray that are building
 blocks for paralleling using ray framework.
"""

# pylint: disable=line-too-long, trailing-whitespace, too-many-lines, too-many-instance-attributes, too-few-public-methods

import functools
import time

# ray can only be initialized once!!
# inside one thread, ray can only be initialized once
# and also for many threads, ray can only be initialized once in the main thread! not all the thread once!!
# even we use cache, it will only cache the current thread.
@functools.lru_cache(None)
def get_ray(num_cpus):
    """
    Obtain ray object.
    If ray is initialized, this will directly return the ray object;
    otherwise it will initialize ray according to user-specified number of cpus.
    """
    import ray  # pylint: disable=import-outside-toplevel
    #print(ray.is_initialized())
    if not ray.is_initialized():
        # initialize N cpus according to user's input
        if num_cpus > 1:
            ray.init(num_cpus=num_cpus)
        # if user's input is True, then use all the cpus
        elif num_cpus is True:
            ray.init()
        # make sure ray is initialized
        if not ray.is_initialized():
            raise ValueError("ray initialization failed!!")

    return ray

def read_out_ray(jobs_list):
    r'''
    Retrieve result from ray.
    '''
    #print("read_out_ray start")
    import ray  # pylint: disable=import-outside-toplevel
    if not ray.is_initialized():
        raise ValueError("ray should be initialized before!!!")
    #print("read_out_ray get ray")
    while True:
        for i, _ in enumerate(jobs_list):
            job = jobs_list[i]
            #print("read_out_ray job")
            done = bool(ray.wait([job], timeout=0)[0])
            #print("read_out_ray job done")
            if done:
                del jobs_list[i]
                #print("read_out_ray ray get start")
                trial = ray.get(job)
                #print("read_out_ray ray get end")
                return trial
        time.sleep(2.e-6)
