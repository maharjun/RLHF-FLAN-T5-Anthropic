from __future__ import annotations
import torch
from typing import List, Tuple, overload, Mapping
from dataclasses import fields, is_dataclass
import inspect
from collections import OrderedDict

from synth_orig_disc.utils.hydrashim import DictConfig
from synth_orig_disc.utils.looputils import get_loop_interval, LoopInterval
from synth_orig_disc.utils.data.loader import random_batch_dataset
from abc import ABC, abstractmethod

def _have_same_parameters(func1, func2):
    func1_params = inspect.signature(func1).parameters
    func2_params = inspect.signature(func2).parameters

    if len(func1_params) != len(func2_params):
        return False

    return all(x.name == y.name and x.kind == y.kind and x.default == y.default
               for (x, y) in zip(func1_params.values(), func2_params.values()))

def _sample_action_with_trained_model(self, loop_context):
    ...

def _sample_periodic_action(self, loop_context, n_batches_since_last):
    ...

def _sample_periodic_action_at_end(self, loop_context, n_batches_since_last, at_end=False):
    ...

@overload
def periodic_action(func):
    ...

@overload
def periodic_action(do_at_end=False):
    ...

def periodic_action(func=None, *, do_at_end=False):
    def inner_dec(func):
        if do_at_end:
            if not _have_same_parameters(func, _sample_periodic_action_at_end):
                raise TypeError("periodic_action: if do_at_end=True, then function signature should match"
                                " the following: _sample_periodic_action_at_end(loop_context, n_batches_since_last, at_end=False)")
        else:
            if not _have_same_parameters(func, _sample_periodic_action):
                raise TypeError("periodic_action: periodic action should have the following signature:"
                                " _sample_periodic_action(loop_context, n_batches_since_last)")

        func._is_training_loop_periodic_action = True
        func._is_training_loop_end_action = do_at_end
        return func

    if func is None:
        return inner_dec
    else:
        return inner_dec(func)


def with_trained_model(func):
    if not _have_same_parameters(func, _sample_action_with_trained_model):
        raise TypeError("with_trained_model: The function signature should match"
                        " the following: _sample_action_with_trained_model(loop_context)")

    func._is_training_loop_trained_model_action = True
    return func


class TrainingLoop(ABC):

    def __init__(self, n_epochs, train_batch_size, eval_batch_size, loop_intervals: DictConfig, rng: torch.Generator):
        self.n_epochs = n_epochs
        self.train_batch_size = train_batch_size
        self.eval_batch_size = eval_batch_size
        self.rng = rng

        self.periodic_action_intervals = {**loop_intervals}
        for pa_name in self.__class__._periodic_actions:
            if pa_name not in self.periodic_action_intervals:
                raise AttributeError(f"TrainingLoop.__init__(): loop_params.intervals does not contain interval parameters for periodic action {pa_name}")

        self._base_loop_initialized = True

    @abstractmethod
    def init_loop_context(self, *args, **kwargs):
        ...

    @abstractmethod
    def train_step(self, batch_dataset_struct, loop_context):
        ...

    @abstractmethod
    def load_best_model(self, loop_context):
        ...

    @abstractmethod
    def to_stop(self, loop_context):
        ...


    def __init_subclass__(cls):
        if not hasattr(cls, '_periodic_actions'):
            cls._periodic_actions = []
        if not hasattr(cls, '_end_actions'):
            cls._end_actions = []
        if not hasattr(cls, '_trained_model_actions'):
            cls._trained_model_actions = []

        cls._periodic_actions.extend(k for k, v in cls.__dict__.items() if getattr(v, "_is_training_loop_periodic_action", False))
        cls._end_actions.extend(k for k, v in cls.__dict__.items() if getattr(v, "_is_training_loop_end_action", False))
        cls._trained_model_actions.extend(k for k, v in cls.__dict__.items() if getattr(v, "_is_training_loop_trained_model_action", False))

    def _verify_loop_context(self, loop_context):
        if not is_dataclass(loop_context):
            raise TypeError("TrainingLoop._verify_loop_context: The loop_context object must be a dataclass")

        loop_context_fieldnames = [f.name for f in fields(loop_context)]
        required_loop_context_members = ('train_data', 'batch_counter', 'epoch_counter')
        if not all(x in loop_context_fieldnames for x in required_loop_context_members):
            raise TypeError(f"TrainingLoop._verify_loop_context: The loop_context dataclass"
                            f" {type(loop_context).__name__} must contain the members {required_loop_context_members}")

    def train_loop(self, loop_context):
        if not self._base_loop_initialized:
            raise RuntimeError("TrainingLoop.train_loop cannot be run without initializing TrainingLoop. (did you forget to use super().__init__?)")

        self._verify_loop_context(loop_context)
        n_train = len(loop_context.train_data)

        periodic_intervals: List[Tuple[str, LoopInterval]] = [
            (action_name, get_loop_interval(n_train, self.train_batch_size, **self.periodic_action_intervals[action_name], split_across_epochs=False))
            for action_name in self.__class__._periodic_actions]

        end_intervals = [
            (action_name, loop_interval)
            for action_name, loop_interval in periodic_intervals
            if action_name in self.__class__._end_actions]

        train_data_loader = random_batch_dataset(dataset=loop_context.train_data, batch_size=self.train_batch_size,
                                                 generator=self.rng)

        for datastruct, epoch_ended in train_data_loader:
            self.train_step(datastruct, loop_context)

            for action_name, loop_interval in periodic_intervals:
                if loop_interval.is_interval_complete():
                    n_batches_since_last = loop_interval.iters_in_previous_interval()
                    getattr(self, action_name)(loop_context, n_batches_since_last)

            loop_context.batch_counter += 1
            if epoch_ended:
                # logger.info(f"Finished Epoch {self.epoch_counter}")
                # logger.info("====================================================")
                loop_context.epoch_counter += 1
                loop_context.batch_counter = 0

            if self.to_stop(loop_context) or loop_context.epoch_counter == self.n_epochs:
                break

        for action_name, loop_interval in end_intervals:
            n_batches_since_last = loop_interval.iters_since_last_interval()
            if n_batches_since_last > 0:
                getattr(self, action_name)(loop_context, n_batches_since_last, at_end=True)

        self.load_best_model(loop_context)
        for action_name in self.__class__._trained_model_actions:
            getattr(self, action_name)(loop_context)