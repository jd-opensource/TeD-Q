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
This module contains the :class:`StorageBase` abstract base class and
 :class:`CircuitStorage` class for storing context. :class:`OperatorBase`
 and :class:`measurement`
"""

# pylint: disable=line-too-long, trailing-whitespace, too-many-lines, too-many-instance-attributes, too-few-public-methods

import abc
from tedq.global_variables import GlobalVariables


class StorageBase(abc.ABC):
    r"""
    Abstract base class for storing information by using context manager
    method.

    """
    _active_warehouse = GlobalVariables.get_value("global_deque")
    # The stack of contexts that are currently active.
    def __enter__(self):

        """This function will be executed when "with" starts to execute.

        Adds this instance to the global list of active contexts.

        Returns:
            QueuingContext: this instance
        """
        StorageBase._active_warehouse.append(self)

        return self

    def __exit__(self, exception_type, exception_value, traceback):
        """Remove this instance from the global list of active contexts."""
        StorageBase._active_warehouse.pop()

    @abc.abstractmethod
    def _append(self, obj, **kwargs):
        """Append an object to this QueuingContext instance.
        
        Args:
            obj: The object to be appended
        """

    @classmethod
    def recording(cls):
        """Whether a queuing context is active and recording operations"""
        # become False if nothing inside deque()
        return bool(cls._active_warehouse)

    @classmethod
    def active_context(cls):
        """Returns the currently active queuing context."""
        if cls.recording():
            return cls._active_warehouse[-1]

        return None

    @classmethod
    def append(cls, obj, **kwargs):
        """Append an object to the queue(s).
        
        Args:
            obj: the object to be appended
        """
        if cls.recording():
            cls.active_context()._append(  # pylint: disable=protected-access
                obj, **kwargs
            )  # pylint: disable=protected-access

    @abc.abstractmethod
    def _remove(self, obj):
        """Remove an object from this QueuingContext instance.
        
        Args:
            obj: the object to be removed
        """

    @classmethod
    def remove(cls, obj):
        """Remove an object from the queue(s) if it is in the queue(s).
        
        Args:
            obj: the object to be removed
        """
        if cls.recording():
            cls.active_context()._remove(obj)  # pylint: disable=protected-access

    @classmethod
    def update_info(cls, obj, **kwargs):
        """Updates information of an object in the active queue.
        
        Args:
            obj: the object with metadata to be updated
        """
        if cls.recording():
            cls.active_context()._update_info(  # pylint: disable=protected-access
                obj, **kwargs
            )  # pylint: disable=protected-access

    def _update_info(self, obj, **kwargs):
        """Updates information of an object in the queue instance."""
        raise NotImplementedError

    @classmethod
    def get_info(cls, obj):
        """Retrieves information of an object in the active queue.
        
        Args:
            obj: the object with metadata to be retrieved
        
        Returns:
            object metadata
        """
        if cls.recording():
            return cls.active_context()._get_info(  # pylint: disable=protected-access
                obj
            )  # pylint: disable=protected-access

        return None

    def _get_info(self, obj):
        """Retrieves information of an object in the queue instance."""
        raise NotImplementedError


class CircuitStorage(StorageBase):
    r"""
    CircuitStorage
    """
    def __init__(self):
        r"""
        Initialize a circuit storage
        """
        self.storage_context = []

    def _append(self, obj, **kwargs):
        r"""
        """
        self.storage_context.append(obj)
        # print("MyQueue  put into list")

    def _remove(self, obj):
        r"""
        """
        self.storage_context.remove(obj)
