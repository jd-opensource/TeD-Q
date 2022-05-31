'''
Initialize global variables
'''
from collections import deque
from tedq.global_variables import GlobalVariables

GlobalVariables.set_value("global_deque", deque())
