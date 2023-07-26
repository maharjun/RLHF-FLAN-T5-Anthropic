from __future__ import annotations
from typing import Type
from dataclasses import is_dataclass
from abc import ABC, abstractmethod
import inspect

@abstractmethod
def output(self, batch_dataset_struct):
    ...


class Accumulator(ABC):
    OutputType: Type = None

    @abstractmethod
    def update_state(self, stage_outputs):
        ...

    @abstractmethod
    def reset(self):
        ...

    def update(self, stage_outputs):
        if not isinstance(stage_outputs, self.__class__.OutputType):
            raise TypeError(f"Accumulator.update: Class stage_outputs should be of type OutputType"
                            f" ({self.__class__.OutputType.__name__}), got type {type(stage_outputs)} instead")
        return self.update_state(stage_outputs)

    def __init_subclass__(cls):
        if not inspect.isabstract(cls):
            OutputType = getattr(cls, 'OutputType', None)
            if OutputType is None:
                raise TypeError(f"{Accumulator.__name__}.__init_subclass__: Class {cls.__name__} inheriting from Accumulator, must specify the class variable {'OutputType'}")
            if not isinstance(OutputType, type):
                raise TypeError(f"{Accumulator.__name__}.__init_subclass__: Class {cls.__name__}.{'OutputType'} should be a type, got {OutputType} instead")
            if not is_dataclass(getattr(cls, 'OutputType', None)):
                raise TypeError(f"{Accumulator.__name__}.__init_subclass__: Class {cls.__name__}.{'OutputType'} must be a dataclass type, Currently got {OutputType.__name__}")
