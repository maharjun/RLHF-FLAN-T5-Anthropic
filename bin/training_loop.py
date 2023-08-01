import logging
import os
from typing import Optional
from dataclasses import dataclass
from collections import deque
from abc import abstractmethod, ABC
import torch
from tqdm import tqdm
from contextlib import ExitStack

from simmanager.tools import Timer

from datasets import DatasetDict, Dataset
from rlhf_flant5.utils.basicutils import center_msg
from rlhf_flant5.utils.hydrashim import DictConfig
from rlhf_flant5.utils.dillshim import dill

# from rlhf_flant5.utils.data.loader import random_batch_dataset
from rlhf_flant5.utils.torchutils import random_seed
from rlhf_flant5.utils.torchutils import copy_model_to_cpu
from rlhf_flant5.utils.looputils import get_loop_interval, LoopInterval, NeverLoopInterval

from rlhf_flant5.utils.data.loader import random_batch_dataset
from rlhf_flant5.utils.data.loader import batch_indices_for_one_epoch

@dataclass
class GlobalLogInfo:
    global_step: int


class MainModule(torch.nn.Module):
    @dataclass
    class Output:
        pass

    # def __init__(self, module_params: DictConfig):
    #     pass

    # @abstractmethod
    # def forward(self, batch_dataset_struct):
    #     pass

    @abstractmethod
    def backward(self, module_output):
        pass


class Tracker(ABC):
    # def __init__(self, logger, stage_name):
    #     self.logger = logger
    #     self.stage_name = stage_name

    def attach_global_info(self, global_info: GlobalLogInfo):
        self.global_info = global_info
        for x in self.__dict__.values():
            if isinstance(x, Tracker):
                x.attach_global_info(global_info)

    @abstractmethod
    def update(self, module_output, time=None):
        ...

    @abstractmethod
    def reset(self):
        ...

    @abstractmethod
    def log(self):
        ...


class TimedTracker(Tracker):
    def __init__(self):
        pass

    def update_time(self, time):
        if not isinstance(time, float):
            raise TypeError("TimedTracker.update_time, Expected float for time")

        if not hasattr(self, 'total_time'):
            self.total_time = time
        else:
            self.total_time += time

    def reset_time(self):
        self.total_time = 0


class ValidationTracker(Tracker):
    # def __init__(self, logger, stage_name):
    #     self.logger = logger
    #     self.stage_name = stage_name
    @abstractmethod
    def validation_loss(self, module_output: MainModule.Output):
        ...


class Checkpointer:
    def __init__(self, output_filepath):
        self.output_filepath = output_filepath

    def save_model(self, model):
        temp_filepath = self.output_filepath + '.temp'
        with open(temp_filepath, 'wb') as fout:
            dill.dump(model, fout)
        os.rename(temp_filepath, self.output_filepath)


class ModelQueue:
    def __init__(self, queue_size):
        self.model_queue = deque([None]*queue_size, maxlen=queue_size)
        self.val_loss_queue = deque([float('inf')]*queue_size, maxlen=queue_size)

    def update(self, main_module: MainModule, val_tracker: Tracker):
        self.model_queue.append(copy_model_to_cpu(main_module).state_dict())
        self.val_loss_queue.append(val_tracker.validation_loss())

    def load_best_module_into(self, main_module: MainModule):
        min_eval_loss_queue = min(self.val_loss_queue)
        min_eval_loss_queue_ind = self.val_loss_queue.index(min_eval_loss_queue)
        main_module.load_state_dict(self.model_queue[min_eval_loss_queue_ind])

    def no_improvement(self):
        return self.val_loss_queue.index(min(self.val_loss_queue)) == 0


def init_train_val_rngs(data_rng: torch.Generator, logger: logging.Logger):
    # Initializing random generators
    train_data_seed = random_seed(data_rng).item()
    val_data_seed = random_seed(data_rng).item()
    train_rng = torch.Generator(data_rng.device)
    val_rng = torch.Generator(data_rng.device)
    train_rng.manual_seed(train_data_seed)
    val_rng.manual_seed(val_data_seed)

    logger.info(f"Train data seed: {train_data_seed}")
    logger.info(f"Validation data seed: {val_data_seed}")
    return train_rng, val_rng


def log_periodic_action(logger, loop_interval :LoopInterval, action_name: str, epoch_counter, batch_counter, final_action: bool = False):
    if not final_action:
        msg = (f"Performing {action_name} after {loop_interval.iters_in_previous_interval()}"
               f" batches in Epoch: {epoch_counter}, Batch: {batch_counter}")
    else:
        msg = (f"Performing {action_name} after {loop_interval.iters_since_last_interval()}"
               f" batches in Epoch: {epoch_counter}, Batch: {batch_counter}")

    logger.info(center_msg(msg, total_len=100, padding_char='='))


def train_status_message(train_tracker):
    train_tracker.log()
    train_tracker.reset()
    train_tracker.reset_time()


def evaluation(stage_name: str, main_module: MainModule, dataset: Dataset, batch_size, data_rng, tracker, logger,
               n_data_to_run: Optional[int] = None,
               model_queue: Optional[ModelQueue] = None,
               show_progress: bool = False):

    main_module.eval()

    if n_data_to_run is None:
        n_data_to_run = len(dataset)
        data_rng = None  # No shuffling since we're running on the entire dataset
    else:
        n_data_to_run = min(n_data_to_run, len(dataset))
    n_batches = (n_data_to_run + batch_size - 1) // batch_size

    data_inds_loader = batch_indices_for_one_epoch(len(dataset), batch_size, data_rng, device=dataset.device)
    
    timer = Timer(logger, 'INFO')
    with timer(f"Performing {stage_name} for {n_data_to_run} samples"):
        with ExitStack() as E:
            E.enter_context(torch.no_grad())
            if show_progress:
                progress_tracker = E.enter_context(tqdm(data_inds_loader, unit='points', total=n_data_to_run))

            for i, val_batch_inds in tqdm(enumerate(data_inds_loader)):
                val_batch_dataset_struct = dataset[val_batch_inds]
                if show_progress:
                    progress_tracker.update(len(val_batch_inds))
                tracker.update(main_module(val_batch_dataset_struct))
                if i >= n_batches: break
        tracker.log()
        if model_queue is not None:
            model_queue.update(main_module, tracker)

    tracker.reset()

def verify_tracker(tracker_name, tracker, is_validation, is_timed):

    required_methods = ('update', 'reset', 'log')
    if is_validation:
        required_methods += ('validation_loss',)
    if is_timed:
        required_methods += ('update_time', 'reset_time')

    msg = f"{tracker_name}, of type {type(tracker)} should implement the {', '.join(required_methods)} methods"

    missing_methods = []
    for methname in required_methods:
        if not (hasattr(tracker, methname) and callable(getattr(tracker, methname))):
            missing_methods.append(methname)

    if missing_methods:
        raise TypeError(f"{msg}. Missing methods {{{', '.join(missing_methods)}}}")


def run_train_eval_loop(dataset: DatasetDict, main_module: MainModule, optimizer: torch.optim.Optimizer,
                        loop_params: DictConfig, data_rng: torch.Generator,
                        train_tracker: TimedTracker, val_tracker: ValidationTracker, test_tracker: Tracker,
                        logger: logging.Logger,
                        checkpointer: Checkpointer,
                        scheduler: Optional[torch.optim.lr_scheduler.LRScheduler] = None,
                        progress_tracker: Optional[Tracker] = None):

    verify_tracker('train_tracker', train_tracker, is_validation=False, is_timed=True)
    verify_tracker('val_tracker', val_tracker, is_validation=True, is_timed=False)
    verify_tracker('test_tracker', test_tracker, is_validation=True, is_timed=False)
    train_timer = Timer(logger=None)
    timer = Timer(logger=logger, log_level='INFO')

    train_rng, val_rng = init_train_val_rngs(data_rng, logger)

    train_dataset = dataset['train']
    val_dataset = dataset['val']
    test_dataset = dataset['test']

    # Initializing local vars from config
    train_batch_size = loop_params.train_batch_size
    val_batch_size = loop_params.val_batch_size
    n_epochs = loop_params.n_epochs
    n_batches = loop_params.n_batches

    # round up to the nearest validation batch
    n_val_data = loop_params.n_val_data

    # Initialize training tracker
    model_queue = ModelQueue(loop_params.max_non_optim_vals+1)

    # Initiate loop interval counters
    validation_LI = get_loop_interval(len(train_dataset), train_batch_size, **loop_params.intervals.validation, split_across_epochs=False)
    train_status_update_LI = get_loop_interval(len(train_dataset), train_batch_size, **loop_params.intervals.train_status_update, split_across_epochs=False)
    checkpointing_LI = get_loop_interval(len(train_dataset), train_batch_size, **loop_params.intervals.checkpointing, split_across_epochs=False)
    progress_logging_LI = NeverLoopInterval()
    if progress_tracker is not None:
        progress_logging_LI = get_loop_interval(len(train_dataset), train_batch_size, **loop_params.intervals.progress_logging)


    global_info = GlobalLogInfo(global_step=0)
    train_tracker.attach_global_info(global_info)
    val_tracker.attach_global_info(global_info)
    test_tracker.attach_global_info(global_info)
    if progress_tracker is not None:
        progress_tracker.attach_global_info(global_info)

    epoch_counter = 0
    batch_counter = 0
    sample_counter = 0
    global_info.global_step = 0

    optimizer.zero_grad(set_to_none=True)

    # train_data_loader = torch.utils.data.DataLoader(dataset=train_dataset,
    #                                                 batch_size=train_batch_size, 
    #                                                 shuffle=True,
    #                                                 num_workers=2,
    #                                                 prefetch_factor=4,
    #                                                 persistent_workers=True)
    # val_data_loader = torch.utils.data.DataLoader(dataset=val_dataset,
    #                                               batch_size=val_batch_size, 
    #                                               shuffle=True,
    #                                               num_workers=2,
    #                                               prefetch_factor=4,
    #                                               persistent_workers=True)

    train_data_loader = random_batch_dataset(train_dataset, train_batch_size, train_rng)
    for batch_dataset_struct, _ in train_data_loader:
        with train_timer("Training Timing"):
            main_module.train()
            module_output: MainModule.Output = main_module(batch_dataset_struct)
            main_module.backward(module_output)

        # Update train metrics etc
        with torch.no_grad():
            train_tracker.update(module_output)
            train_tracker.update_time(train_timer.profile_list[-1][1])
            if progress_tracker is not None:
                progress_tracker.update(module_output)

        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

        if train_status_update_LI.is_interval_complete():
            log_periodic_action(logger, train_status_update_LI, 'Status Update', epoch_counter, batch_counter)
            train_status_message(train_tracker)  # Also resets tracker

        if validation_LI.is_interval_complete():
            log_periodic_action(logger, progress_logging_LI, 'Validation', epoch_counter, batch_counter)
            evaluation(stage_name='Validation',
                       main_module=main_module,
                       dataset=val_dataset,
                       batch_size=val_batch_size,
                       data_rng=val_rng,
                       tracker=val_tracker,
                       logger=logger,
                       n_data_to_run=n_val_data,
                       model_queue=model_queue)

        if progress_logging_LI.is_interval_complete():
            log_periodic_action(logger, progress_logging_LI, 'Logging Progress', epoch_counter, batch_counter)
            progress_tracker.log()
            progress_tracker.reset()

        if checkpointing_LI.is_interval_complete():
            log_periodic_action(logger, checkpointing_LI, 'Saving Checkpoint', epoch_counter, batch_counter)
            with timer("Saving Model Checkpoint"):
                checkpointer.save_model(main_module)

        batch_counter += 1
        sample_counter += train_batch_size
        global_info.global_step += 1
        if sample_counter >= len(train_dataset):
            batch_counter = 0
            sample_counter -= len(train_dataset)
            epoch_counter += 1

            if scheduler is not None:
                scheduler.step()

        if epoch_counter >= n_epochs or \
           (epoch_counter == n_epochs and batch_counter <= n_batches) \
           or model_queue.no_improvement():
            break

    if train_status_update_LI.iters_since_last_interval() > 0:
        log_periodic_action(logger, train_status_update_LI, 'Status Update', epoch_counter, batch_counter, final_action=True)
        train_status_message(train_tracker)

    if validation_LI.iters_since_last_interval() > 0:
        # Note that the above condition also means that the loop has not quit due to no improvement
        log_periodic_action(logger, progress_logging_LI, 'Validation', epoch_counter, batch_counter)
        evaluation(stage_name='Validation',
                   main_module=main_module,
                   dataset=val_dataset,
                   batch_size=val_batch_size,
                   data_rng=val_rng,
                   tracker=val_tracker,
                   logger=logger,
                   model_queue=model_queue,
                   n_data_to_run=n_val_data)

    # load the best module into main_module
    model_queue.load_best_module_into(main_module)

    # perform testing
    logger.info(f"Performing Final Testing on {len(test_dataset)} points with best model")
    evaluation(stage_name='Testing',
               main_module=main_module,
               dataset=test_dataset,
               batch_size=val_batch_size,
               data_rng=None,  # No shuffling
               tracker=test_tracker,
               logger=logger,
               show_progress=True)

    return main_module
