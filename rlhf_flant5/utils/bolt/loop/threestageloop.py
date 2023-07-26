from __future__ import annotations
import torch
from collections import deque
from typing import Any, Mapping, List
from dataclasses import dataclass, field
import copy
from abc import ABC, abstractmethod

from simmanager.tools import Timer

from synth_orig_disc.utils.looputils import LoopInterval
from synth_orig_disc.utils.hydrashim import DictConfig
from synth_orig_disc.utils.data.loader import batch_dataset_for_one_epoch

from synth_orig_disc.utils.bolt.loop import TrainingLoop, periodic_action, with_trained_model
from synth_orig_disc.utils.bolt.accumulators import Accumulator, Loggable

# Abstract classes used for type indication
class _LoggableAccumulator(Accumulator, Loggable):
    def __init_subclass__(cls, subclass):
        mixin_names = [x.name for x in cls.__bases__ if x.name.split('.')[-1] != 'Accumulator']
        raise TypeError(f"Cannot inherit from {cls.__name__}, inherit directly from {Accumulator.__name__}, with interfaces {', '.join(mixin_names)} instead")


class ThreeStageMainModule(torch.nn.Module, ABC):

    @abstractmethod
    def verify_dataset(self, dataset):
        ...

    @abstractmethod
    def train_output(self, batch_dataset_struct):
        ...

    @abstractmethod
    def val_output(self, batch_dataset_struct):
        ...

    @abstractmethod
    def eval_output(self, batch_dataset_struct):
        ...

    @abstractmethod
    def backward(self, train_output):
        ...

    @abstractmethod
    def validation_loss(self, validation_accum):
        ...


class ThreeStageTrainingLoop(TrainingLoop):

    @dataclass
    class LoopContext:
        train_data: Any = field(default=None)
        val_data: Any = field(default=None)
        eval_data: Any = field(default=None)

        epoch_counter: int = field(default=0)
        batch_counter: int = field(default=0)

        model_queue: deque = field(default=None)
        val_loss_queue: deque = field(default=None)

        train_accum: _LoggableAccumulator = field(default=None)
        val_accum: _LoggableAccumulator = field(default=None)
        eval_accum: _LoggableAccumulator = field(default=None)

        other_train_accums: Mapping[str, Accumulator] = field(default_factory=dict)
        other_val_accums: Mapping[str, Accumulator] = field(default_factory=dict)
        other_eval_accums: Mapping[str, Accumulator] = field(default_factory=dict)

        other_accums: Mapping[str, Accumulator] = field(default_factory=dict)


    def __init__(self, main_module: ThreeStageMainModule, optimizer, loop_params: DictConfig, rng: torch.Generator, logger: torch.logger, log_prefix=None):
        super().__init__(loop_params.n_epochs, loop_params.train_batch_size, loop_params.eval_batch_size, loop_params.intervals, rng)

        self.main_module = main_module
        self.optimizer = optimizer

        # super init
        self.logger = logger
        self.timer = Timer(logger, 'INFO')
        if log_prefix is None:
            self.log_prefix = ''
        else:
            self.log_prefix = f'{log_prefix} '

        self.max_non_optim_vals = loop_params.max_non_optim_vals
        self.periodic_action_intervals = {**loop_params.intervals}
        for pa_name in self.__class__._periodic_actions:
            if pa_name not in self.periodic_action_intervals:
                raise AttributeError(f"loop_params.intervals does not contain interval parameters for periodic action {pa_name}")

        self.batch_counter = 0

    def register_train_accumulator(self, name, new_train_accum, loop_context: LoopContext):
        if name in loop_context.other_train_accums:
            raise ValueError(f"ThreeStageTrainingLoop.register_train_accumulator: Train accumulator by name {name} already exists")
        loop_context.other_train_accums[name] = new_train_accum

    def register_val_accumulator(self, name, new_val_accum, loop_context: LoopContext):
        if name in loop_context.other_val_accums:
            raise ValueError(f"ThreeStageTrainingLoop.register_val_accumulator: Validation accumulator by name {name} already exists")
        loop_context.other_val_accums[name] = new_val_accum

    def register_eval_accumulator(self, name, new_eval_accum, loop_context: LoopContext):
        if name in loop_context.other_eval_accums:
            raise ValueError(f"ThreeStageTrainingLoop.register_eval_accumulator: Evaluation accumulator by name {name} already exists")
        loop_context.other_eval_accums[name] = new_eval_accum

    ##############################################################
    # These methods must be implemented to extend TrainingLoop
    ##############################################################
    # This function defines the inputs to train_loop
    def init_loop_context(self, train_data, validation_data, evaluation_data,
                          train_accum: _LoggableAccumulator,
                          val_accum: _LoggableAccumulator,
                          eval_accum: _LoggableAccumulator):

        loop_context = self.__class__.LoopContext()

        self.main_module.verify_dataset(train_data)
        self.main_module.verify_dataset(validation_data)
        self.main_module.verify_dataset(evaluation_data)

        loop_context.train_data = train_data
        loop_context.val_data = validation_data
        loop_context.eval_data = evaluation_data

        def verify_stage_type(accum_name, accum, accumtype):
            if not isinstance(accum, accumtype):
                raise TypeError(f"ThreeStageTrainingLoop.init_loop_context: {accum_name} must"
                                f" be a subclass of {accumtype.__name__}, Currently got type {type(accum)}")

        verify_stage_type('train_accum', train_accum, Accumulator)
        verify_stage_type('val_accum', val_accum, Accumulator)
        verify_stage_type('eval_accum', eval_accum, Accumulator)

        verify_stage_type('train_accum', train_accum, Loggable)
        verify_stage_type('val_accum', val_accum, Loggable)
        verify_stage_type('eval_accum', eval_accum, Loggable)

        loop_context.train_accum = train_accum
        loop_context.val_accum = val_accum
        loop_context.eval_accum = eval_accum

        return loop_context


    def train_step(self, batch_dataset_struct, loop_context: LoopContext):
        train_output = self.main_module.train_output(batch_dataset_struct)
        with torch.no_grad():
            loop_context.train_accum.update(train_output)
            for accum in loop_context.other_train_accums.values():
                accum.update(train_output)

        # optimization step
        self.optimizer.zero_grad(set_to_none=True)
        self.main_module.backward(train_output)
        self.optimizer.step()

    def load_best_model(self, loop_context: LoopContext):
        min_eval_loss_queue = min(loop_context.val_loss_queue)
        min_eval_loss_queue_ind = loop_context.val_loss_queue.index(min_eval_loss_queue)
        if loop_context.val_loss_queue[-1] > min_eval_loss_queue:
            self.main_module.load_state_dict(loop_context.model_queue[min_eval_loss_queue_ind])

    def to_stop(self, loop_context: LoopContext):
        return loop_context.val_loss_queue.index(min(loop_context.val_loss_queue)) == 0

    #############
    # Utilities
    #############
    def _run_stage_on_data(self, stage_output_func, stage_accums: List[Accumulator], data):
        complete_data_loader = batch_dataset_for_one_epoch(dataset=data, batch_size=self.eval_batch_size)
        for batch_dataset_struct in complete_data_loader:
            stage_output = stage_output_func(batch_dataset_struct)
            for accum in stage_accums:
                accum.update(stage_output)

    def _get_n_batches_since_last(self, loop_interval: LoopInterval, at_end):
        if at_end:
            n_batches_since_last = loop_interval.iters_since_last_interval()
        else:
            n_batches_since_last = loop_interval.iters_in_previous_interval()
        return n_batches_since_last

    ############################################################################
    # Actions, including evaluation and validation, Note that the validation and
    # evaluation actions are specific to FourStageMainModule, and any number of actions can be added
    ############################################################################

    @periodic_action(do_at_end=True)
    def train_status_update(self, loop_context: LoopContext, n_batches_since_last: int, at_end=False):
        header_message = (f"Status Update after {n_batches_since_last} batches"
                          f" at Epoch {loop_context.epoch_counter},"
                          f" batch {loop_context.batch_counter}")
        self.logger.info(self.log_prefix + header_message)
        loop_context.train_accum.log()
        loop_context.train_accum.reset()

    @periodic_action(do_at_end=True)
    def validation(self, loop_context: LoopContext, n_batches_since_last: int, at_end=False):
        header_message = (f"Validation after {n_batches_since_last} batches"
                          f" at Epoch {loop_context.epoch_counter},"
                          f" batch {loop_context.batch_counter},"
                          f" ({len(loop_context.val_data)} points)")

        with torch.no_grad(), self.timer(self.log_prefix + header_message):
            self.logger.info(self.log_prefix + f"Performing {header_message}")
            self._run_stage_on_data(self.main_module.val_output, 
                                    [loop_context.val_accum, *loop_context.other_val_accums.values()],
                                    loop_context.val_data)
            loop_context.val_accum.log()

            # storing model in queue to pick best model
            if loop_context.model_queue is None:
                loop_context.model_queue = deque([None]*(self.max_non_optim_vals+1), maxlen=self.max_non_optim_vals+1)
                loop_context.val_loss_queue = deque([float('inf')]*(self.max_non_optim_vals+1), maxlen=self.max_non_optim_vals+1)

            loop_context.model_queue.append(copy.deepcopy(self.main_module.state_dict()))
            loop_context.val_loss_queue.append(self.main_module.validation_loss(loop_context.val_accum))
            loop_context.val_accum.reset()

    @with_trained_model
    def evaluation(self, loop_context: LoopContext):
        header_message = (f"Evaluation with best model on {len(loop_context.eval_data)} datapoints"
                          f" at Epoch {loop_context.epoch_counter},"
                          f" batch {loop_context.batch_counter},"
                          f" ({len(loop_context.eval_data)} points)")
        self.logger.info(self.log_prefix + header_message)
        self._run_stage_on_data(self.main_module.eval_output, 
                                [loop_context.eval_accum, *loop_context.other_eval_accums.values()],
                                loop_context.eval_data)
        loop_context.eval_accum.log()
        loop_context.eval_accum.reset()
