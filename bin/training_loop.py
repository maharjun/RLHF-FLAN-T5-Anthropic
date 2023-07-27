import logging
from typing import Optional
from dataclasses import dataclass
from collections import deque
from abc import abstractmethod, ABC
import torch

from simmanager.tools import Timer

from datasets import DatasetDict, Dataset
from rlhf_flant5.utils.basicutils import center_msg
from rlhf_flant5.utils.hydrashim import DictConfig

# from rlhf_flant5.utils.data.loader import random_batch_dataset
from rlhf_flant5.utils.torchutils import random_seed
from rlhf_flant5.utils.torchutils import copy_model_to_cpu
from rlhf_flant5.utils.looputils import get_loop_interval, LoopInterval

from rlhf_flant5.utils.data.loader import random_batch_dataset
from rlhf_flant5.utils.data.loader import batch_dataset_for_one_epoch

logger = logging.getLogger('training_loop')
timer = Timer(logger, 'INFO')


class MainModule(torch.nn.Module):
    @dataclass
    class Output:
        pass

    # def __init__(self, module_params: DictConfig):
    #     pass

    @abstractmethod
    def output(self, batch_dataset_struct):
        pass

    @abstractmethod
    def backward(self, module_output: Output):
        pass


class Tracker(ABC):
    # def __init__(self, logger, stage_name):
    #     self.logger = logger
    #     self.stage_name = stage_name

    @abstractmethod
    def update(self, module_output: MainModule.Output):
        ...

    @abstractmethod
    def reset(self):
        ...

    @abstractmethod
    def log(self):
        ...

class ValidationTracker(Tracker):
    # def __init__(self, logger, stage_name):
    #     self.logger = logger
    #     self.stage_name = stage_name
    @abstractmethod
    def validation_loss(self, module_output: MainModule.Output):
        ...


class ModelQueue:
    def __init__(self, queue_size):
        self.model_queue = deque([None]*(self.max_non_optim_vals+1), maxlen=self.max_non_optim_vals+1)
        self.val_loss_queue = deque([float('inf')]*(self.max_non_optim_vals+1), maxlen=self.max_non_optim_vals+1)

    def update(self, main_module: MainModule, val_tracker: Tracker):
        self.model_queue.append(copy_model_to_cpu(self.main_module).state_dict())
        self.val_loss_queue.append(val_tracker.validation_loss())

    def load_best_module_into(self, main_module: MainModule):
        min_eval_loss_queue = min(self.val_loss_queue)
        min_eval_loss_queue_ind = self.val_loss_queue.index(min_eval_loss_queue)
        self.main_module.load_state_dict(self.model_queue[min_eval_loss_queue_ind])

    def no_improvement(self):
        self.val_loss_queue.index(min(self.val_loss_queue)) == 0


def init_train_val_rngs(data_rng: torch.Generator):
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


def init_optimizer(main_module: MainModule, train_params: DictConfig):
    if train_params.optimizer.type == 'AdamW':
        optimizer = torch.optim.AdamW(main_module.parameters, lr=train_params.optimizer.lr, weight_decay=train_params.optimizer.weight_decay)
    elif main_module.optimizer.type == 'Adam':
        optimizer = torch.optim.Adam(main_module.parameters, lr=train_params.optimizer.lr, weight_decay=train_params.optimizer.weight_decay)
    else:
        raise ValueError("The training.optimizer.type should be one of {'AdamW', or 'Adam'}")
    return optimizer


def log_periodic_action(loop_interval :LoopInterval, action_name: str, epoch_counter, batch_counter, final_action: bool = False):
    if not final_action:
        msg = (f"Performing {action_name} after {loop_interval.iters_in_previous_interval()}"
               f" batches in Epoch: {epoch_counter}, Batch: {batch_counter}")
    else:
        msg = (f"Performing {action_name} after {loop_interval.iters_since_last_interval()}"
               f" batches in Epoch: {epoch_counter}, Batch: {batch_counter}")

    logger.info(center_msg(msg, total_len=100, padding_char='='))


def train_status_message(train_tracker):
    train_tracker.log(logger)
    train_tracker.reset()


def evaluation(stage_name: str, main_module: MainModule, dataset: Dataset, batch_size, data_rng, tracker,
               n_batches: Optional[int] = None,
               model_queue: Optional[ModelQueue] = None):

    n_val_data = min(n_batches*batch_size)
    data_loader = batch_dataset_for_one_epoch(dataset, batch_size, data_rng)
    tracker = Tracker()
    with timer(f"Performing {stage_name} for {n_val_data} samples"):
        for i, val_batch_dataset_struct in enumerate(data_loader):
            tracker.update(main_module.output(val_batch_dataset_struct))
            if i >= n_batches: break
        tracker.log()
        if model_queue is not None:
            model_queue.update(main_module, tracker)

def verify_tracker(tracker_name, tracker, is_validation):
    if is_validation:
        msg = f'{tracker_name}, of type {type(tracker)} should implement the update, reset, log, and validation_loss methods'
    else:
        msg = f'{tracker_name}, of type {type(tracker)} should implement the update, reset, and log methods'

    missing_methods = []
    if not (hasattr(tracker, 'update') and callable(tracker.update)):
        missing_methods.append('update')
    if not (hasattr(tracker, 'reset') and callable(tracker.reset)):
        missing_methods.append('reset')
    if not (hasattr(tracker, 'log') and callable(tracker.log)):
        missing_methods.append('log')
    if is_validation and not (hasattr(tracker, 'validation_loss') and callable(tracker.validation_loss)):
        missing_methods.append('validation_loss')

    if missing_methods:
        raise TypeError(f"{msg}. Missing methods {{{', '.join(missing_methods)}}}")


def run_train_eval_loop(train_params: DictConfig, dataset: DatasetDict, main_module: MainModule, data_rng: torch.Generator,
                        train_tracker: Tracker, val_tracker: ValidationTracker, test_tracker: Tracker):

    verify_tracker('train_tracker', train_tracker, is_validation=False)
    verify_tracker('val_tracker', val_tracker, is_validation=True)
    verify_tracker('test_tracker', test_tracker, is_validation=True)

    train_rng, val_rng = init_train_val_rngs(data_rng)

    train_dataset = dataset['train']
    val_dataset = dataset['val']
    test_dataset = dataset['test']

    # Initializing local vars from config
    train_batch_size = train_params.loop.train_batch_size
    val_batch_size = train_params.loop.val_batch_size

    # round up to the nearest validation batch
    n_val_data = train_params.n_val_data
    n_val_data = min(n_val_data, len(val_dataset))
    n_val_batches = (n_val_data + val_batch_size - 1) // val_batch_size

    loop_params = train_params.loop

    # Initialize training tracker
    model_queue = ModelQueue(train_params.max_non_optim_vals+1)

    # Initiate loop interval counters
    validation_LC = get_loop_interval(len(train_dataset), train_batch_size, **loop_params.intervals.validation, split_across_epochs=False)
    train_status_update_LC = get_loop_interval(len(train_dataset), train_batch_size, **loop_params.intervals.train_status_update, split_across_epochs=False)
    
    epoch_counter = 0
    batch_counter = 0
    sample_counter = 0

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

    optimizer = init_optimizer(main_module, train_params)

    train_data_loader = random_batch_dataset(train_dataset, train_batch_size, train_rng)
    for batch_dataset_struct, _ in train_data_loader:
        main_module.train()
        module_output: MainModule.Output = main_module.output(batch_dataset_struct)
        main_module.backward(module_output)

        # Update train metrics etc
        with torch.no_grad():
            train_tracker.update(module_output)

        optimizer.zero_grad(set_to_none=True)
        optimizer.step()

        if train_status_update_LC.is_interval_complete():
            log_periodic_action(train_status_update_LC, 'Status Update', epoch_counter, batch_counter)
            train_status_message(train_tracker)  # Also resets tracker

        if validation_LC.is_interval_complete():
            evaluation(stage_name='Validation',
                       main_module=main_module,
                       dataset=val_dataset,
                       data_rng=val_rng,
                       tracker=val_tracker,
                       n_batches=n_val_batches,
                       model_queue=model_queue)

        batch_counter += 1
        sample_counter += train_batch_size
        if sample_counter >= len(train_dataset):
            batch_counter = 0
            sample_counter -= len(train_dataset)
            epoch_counter += 1

        if epoch_counter >= train_params.n_epochs or model_queue.no_improvement():
            break

    if train_status_update_LC.iters_since_last_interval() > 0:
        log_periodic_action(train_status_update_LC, 'Status Update', epoch_counter, batch_counter, final_action=True)
        train_status_message(train_tracker)

    if validation_LC.iters_since_last_interval() > 0:
        # Note that the above condition also means that the loop has not quit due to no improvement
        evaluation(stage_name='Validation',
                   main_module=main_module,
                   dataset=val_dataset,
                   data_rng=val_rng,
                   tracker=val_tracker,
                   n_batches=n_val_batches,
                   model_queue=model_queue)

    # load the best module into main_module
    model_queue.load_best_module_into(main_module)

    # perform testing
    logger.info(f"Performing Final Testing on {len(test_dataset)} points")
    evaluation(stage_name='Testing',
               main_module=main_module,
               dataset=test_dataset,
               data_rng=val_rng,
               tracker=test_tracker)