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
This module contains Panel and Linechart for showing training progress.
'''

# pylint: disable=line-too-long, trailing-whitespace, too-many-lines, too-many-instance-attributes, too-few-public-methods, too-many-arguments

import panel as pn
from IPython import display as idisp
import matplotlib.pyplot as plt


class Panel():
    '''
    draw panel of accuracy
    '''
    def __init__(self, name='Accuracy', value=0, bounds=(0,100)):
        pn.extension('echarts')
        option = {"animationDurationUpdate":100}
        self.board = pn.indicators.Gauge(name=name, value=value,
             bounds=bounds, custom_opts = option
        )

    def show(self):
        '''
        show the panel
        '''
        idisp(self.board)  # pylint: disable=not-callable

    def update(self, accurracy):
        '''
        update value
        '''
        _x = round(accurracy,2)
        self.board.value = _x



class Linechart():
    '''
    draw accuracy vs time
    '''
    def __init__(self, xlim=None, ylim=(0,1), xlabel='time (s)',
         ylabel='Accuracy', title='validation accuracy'
    ):
        self.xlim=xlim
        self.ylim=ylim
        self.xlabel=xlabel
        self.ylabel=ylabel
        self.title=title
        self._xx=[]
        self._yy=[]

    def update(self, time, accuracy):
        '''
        update point
        '''
        plt.cla()
        self._xx.append(time)
        self._yy.append(accuracy)
        xuplim = 10*time/len(self._xx) if len(self._xx)<10 else time
        self.xlim=(self._xx[0],xuplim)
        plt.xlim(self.xlim)
        plt.ylim(self.ylim)
        #print(self.xx)

        plt.plot(self._xx, self._yy, color='black')
        plt.scatter(self._xx, self._yy, color='red', s=20)
        plt.title(self.title)
        plt.xlabel(self.xlabel)
        plt.ylabel(self.ylabel)
        idisp.clear_output(wait=True)
        plt.pause(0.00000001)

    @classmethod
    def show(cls):
        '''
        draw
        '''
        plt.show()
