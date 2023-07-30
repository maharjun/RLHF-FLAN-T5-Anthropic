import os
import logging
from dataclasses import dataclass
from math import pow
from typing import Optional

import torch
import numpy as np
from torch.utils.tensorboard.writer import SummaryWriter

from simmanager.tools import Timer
from transformers import T5Model, AutoTokenizer, AutoConfig
from transformers.models.t5.modeling_t5 import T5ForConditionalGeneration
from transformers.models.t5.tokenization_t5_fast import T5TokenizerFast
from transformers.models.t5.configuration_t5 import T5Config

from rlhf_flant5.utils.basicutils import getFrameDir
from rlhf_flant5.utils.hydrashim import DictConfig

# from rlhf_flant5.utils.data.loader import random_batch_dataset
from rlhf_flant5.utils.torchutils import random_seed
from rlhf_flant5.models.rewardmodels import RewardModelNameMap
from rlhf_flant5.models.basemodels import PretrainedEmbeddingsModel

from training_loop import MainModule
from training_loop import Tracker
from training_loop import ValidationTracker
from training_loop import TimedTracker

from torchmetrics.aggregation import MeanMetric
from torchmetrics.classification import BinaryAccuracy
from datasets import load_dataset, DatasetDict, Dataset
from datasets import Features
from datasets import Array2D, Array3D, Array4D, Array5D

def get_float_feature_from_sample_array(array: torch.Tensor):
    if array.ndim <= 2:
        return None

    extra_dims = array.ndim - 1
    if extra_dims == 2:
        return Array2D(array.shape[1:], 'float32')
    if extra_dims == 3:
        return Array3D(array.shape[1:], 'float32')
    if extra_dims == 4:
        return Array4D(array.shape[1:], 'float32')
    if extra_dims == 5:
        return Array5D(array.shape[1:], 'float32')

logger = logging.getLogger('train_reward_model')
timer = Timer(logger, 'INFO')

script_dir = getFrameDir()

@dataclass
class RewardMainOutput:
    chosen_rewards: torch.Tensor
    rejected_rewards: torch.Tensor
    loss: torch.Tensor


class RewardMainModule(MainModule):
    Output = RewardMainOutput

    def __init__(self, model_params: DictConfig, device: torch.device = None, multi_gpu=False):
        super().__init__()

        pretrained_transformer: T5ForConditionalGeneration = \
            T5Model.from_pretrained(model_params.transformer_name) 
        # pretrained_transformer.gradient_checkpointing = True
        # AutoModelForSeq2SeqLM.from_pretrained(model_params.transformer_name) 

        # Disable training for pretrained_encoder
        # pretrained_encoder: T5Stack = pretrained_transformer.encoder
        if not model_params.train_transformer:
            for parameter in pretrained_transformer.parameters():
                parameter.requires_grad = False
            
        self.use_pretrained_output = model_params.use_pretrained_output
        if self.use_pretrained_output and model_params.train_transformer:
            raise RuntimeError("use_pretrained_output cannot be True if model_params.train_transformer is True")

        self.tokenizer: T5TokenizerFast = AutoTokenizer.from_pretrained(model_params.transformer_name)
        self.transformer_config: T5Config = AutoConfig.from_pretrained(model_params.transformer_name)
        model_class = RewardModelNameMap[model_params.reward_model_type]
        self.reward_model: torch.nn.Module
        self.reward_model = model_class(pretrained_transformer, **model_params.reward_model,
                                        use_pretrained_output=self.use_pretrained_output)

        if device:
            self.to(device)

        self._multi_gpu = False
        if multi_gpu:
            n_gpus = torch.cuda.device_count()
            assert n_gpus > 0, "No GPU devices found when multi_gpu == True"
            torch_device_list = [torch.device(f'cuda:{i}') for i in range(n_gpus)]

            self.to(torch_device_list[0])
            self.reward_model = torch.nn.DataParallel(self.reward_model, torch_device_list)
            self._multi_gpu = True

    @property
    def device(self):
        if not self._multi_gpu:
            return next(iter(self.reward_model.parameters())).device
        else:
            return next(iter(self.reward_model.module.parameters())).device

    def output(self, batch_dataset_struct):
        if self.use_pretrained_output:
            chosen_input_embeddings = {k: v for k, v in batch_dataset_struct.items() if k.startswith('chosen_')}
            rejected_input_embeddings = {k: v for k, v in batch_dataset_struct.items() if k.startswith('rejected_')}
            # torch.as_tensor(v, dtype=torch.float32, device=self.device)
            if not len(chosen_input_embeddings):
                raise ValueError("It appears that the pretrained embeddings have not been calculated")
            assert len(chosen_input_embeddings) == len(rejected_input_embeddings)
            assert all(torch.is_tensor(x) for x in chosen_input_embeddings.values())
            assert all(torch.is_tensor(x) for x in rejected_input_embeddings.values())

            reward_model = self.reward_model
            if self._multi_gpu:
                reward_model = self.reward_model.module
            chosen_input_embedding_tensors = reward_model.PTE_tensors_from_dict(chosen_input_embeddings, key_prefix='chosen_')
            rejected_input_embedding_tensors = reward_model.PTE_tensors_from_dict(rejected_input_embeddings, key_prefix='rejected_')

            chosen_rewards = self.reward_model(*chosen_input_embedding_tensors)
            rejected_rewards = self.reward_model(*rejected_input_embedding_tensors)
        else:
            chosen_input = batch_dataset_struct['chosen']
            rejected_input = batch_dataset_struct['rejected']

            chosen_input_tokenised = self.tokenizer(chosen_input, padding=True, return_tensors='pt')  # max_len = 512
            rejected_input_tokenised = self.tokenizer(rejected_input, padding=True, return_tensors='pt')  # max_len = 512

            chosen_input_tokenised = chosen_input_tokenised.to(self.device)
            rejected_input_tokenised = rejected_input_tokenised.to(self.device)

            assert len(chosen_input_tokenised.input_ids) == len(rejected_input_tokenised.input_ids) == len(chosen_input) == len(rejected_input)
            chosen_rewards = self.reward_model(chosen_input_tokenised.input_ids, chosen_input_tokenised.attention_mask)
            rejected_rewards = self.reward_model(rejected_input_tokenised.input_ids, rejected_input_tokenised.attention_mask)

        loss = -torch.nn.functional.logsigmoid(chosen_rewards - rejected_rewards)

        return RewardMainOutput(chosen_rewards=chosen_rewards,
                                rejected_rewards=rejected_rewards,
                                loss=loss)

    def backward(self, module_output: RewardMainOutput):
        total_loss = torch.sum(module_output.loss, dim=0)
        assert total_loss.ndim == 0
        total_loss.backward()


class RewardProgressTracker(Tracker):

    def __init__(self, output_folder: str, logger: logging.Logger, stage_name: str, device: torch.device):
        self.logger = logger
        self.stage_name = stage_name

        self.writer = SummaryWriter(output_folder, f"{self.stage_name} Progress")

        self.chosen_rewards_list = []
        self.rejected_rewards_list = []
        self.loss_list = []

    def update(self, module_output: RewardMainOutput):
        self.chosen_rewards_list.append(module_output.chosen_rewards)
        self.rejected_rewards_list.append(module_output.rejected_rewards)
        self.loss_list.append(module_output.loss)

    def reset(self):
        self.chosen_rewards_list = []
        self.rejected_rewards_list = []
        self.loss_list = []

    def log(self):
        if len(self.chosen_rewards_list) > 0:
            global_step = self.global_info.global_step
            self.logger.info(f"{self.stage_name} Outputting Rewards and stuff for global_step {global_step}")

            output_loss = torch.cat(self.loss_list)
            output_chosen_rewards = torch.cat(self.chosen_rewards_list)
            output_rejected_rewards = torch.cat(self.rejected_rewards_list)
            output_rewards_diff = output_chosen_rewards - output_rejected_rewards
            output_rewards_sigmoid = torch.sigmoid(output_rewards_diff)
            self.writer.add_histogram(f"{self.stage_name} Rewards: Chosen", output_chosen_rewards, global_step=global_step)
            self.writer.add_histogram(f"{self.stage_name} Rewards: Rejected", output_rejected_rewards, global_step=global_step)
            self.writer.add_histogram(f"{self.stage_name} Rewards Diff", output_rewards_diff, global_step=global_step)
            self.writer.add_histogram(f"{self.stage_name} Rewards Diff Sigmoid", output_rewards_sigmoid, global_step=global_step)
            self.writer.add_histogram(f"{self.stage_name} Loss", output_loss, global_step=global_step)

    def __del__(self):
        self.writer.close()


class RewardTracker(ValidationTracker, TimedTracker):
    def __init__(self, logger: logging.Logger, stage_name: str, device: torch.device,
                 tensorboard_output_dir: str = None):
        self.logger = logger
        self.stage_name = stage_name

        self.mean_loss = MeanMetric().to(device=device)
        self.mean_accuracy = BinaryAccuracy().to(device=device)
        self.n_points = 0

        if tensorboard_output_dir is not None:
            self.progress_tracker = RewardProgressTracker(tensorboard_output_dir, self.logger, self.stage_name, device)


    def update(self, module_output: RewardMainOutput):
        self.mean_loss.update(module_output.loss)
        self.mean_accuracy.update(torch.sigmoid(module_output.chosen_rewards - module_output.rejected_rewards),
                                  torch.ones_like(module_output.chosen_rewards))
        self.n_points += len(module_output.loss)
        if hasattr(self, 'progress_tracker'):
            self.progress_tracker.update(module_output)

    def reset(self):
        self.mean_loss.reset()
        self.mean_accuracy.reset()
        self.n_points = 0
        if hasattr(self, 'progress_tracker'):
            self.progress_tracker.reset()

    def validation_loss(self):
        return self.mean_loss.compute()

    def log(self):
        if hasattr(self, 'total_time'):
            self.logger.info(f"Total {self.stage_name} time: {self.total_time} s")
            self.logger.info(f"Mean {self.stage_name} time: {self.total_time*1000/self.n_points} ms")
        self.logger.info(f"Mean {self.stage_name} LogSigmoid Loss: {self.mean_loss.compute():.6}")
        self.logger.info(f"Mean {self.stage_name} Accuracy: {self.mean_accuracy.compute():.6}")
        if hasattr(self, 'progress_tracker'):
            self.progress_tracker.log()


def init_optimizer(main_module: MainModule, optimizer_params: DictConfig, n_epochs):
    if optimizer_params.type == 'AdamW':
        optimizer = torch.optim.AdamW(main_module.parameters(), lr=optimizer_params.init_lr, weight_decay=optimizer_params.weight_decay)
    elif optimizer_params.type == 'Adam':
        optimizer = torch.optim.Adam(main_module.parameters(), lr=optimizer_params.init_lr, weight_decay=optimizer_params.weight_decay)
    else:
        raise ValueError("The training.optimizer.type should be one of {'AdamW', or 'Adam'}")

    gamma = pow(optimizer_params.final_lr/optimizer_params.init_lr, (1/max(n_epochs-1, 1)))
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=gamma)
    return optimizer, scheduler


def filter_datasets_inplace_by_max_tokens_(datasets: DatasetDict, tokenizer: T5TokenizerFast, max_tokens: int):
    assert 'val' not in datasets, "filter_datasets_inplace_by_max_tokens_ should only be called before validation is split off"

    orig_n_train = len(datasets['train'])
    orig_n_test = len(datasets['test'])

    def filter_function(input_batch):
        chosen_input = input_batch['chosen']
        rejected_input = input_batch['rejected']
        chosen_input_tokenised = tokenizer(chosen_input, truncation=False, padding=False, return_tensors='np')
        rejected_input_tokenised = tokenizer(rejected_input, truncation=False, padding=False, return_tensors='np')

        return [len(c) <= max_tokens and len(r) <= max_tokens 
                for c, r in zip(chosen_input_tokenised.input_ids, rejected_input_tokenised.input_ids)]

    datasets['train'] = datasets['train'].filter(function=filter_function, with_indices=False, batched=True, batch_size=1000)
    datasets['test'] = datasets['test'].filter(function=filter_function, with_indices=False, batched=True, batch_size=1000)

    logger.info("Filtered Dataset stats:")
    logger.info(f"   Training Dataset  : {len(datasets['train'])} / {orig_n_train} Points")
    logger.info(f"   Testing Dataset   : {len(datasets['test'])} / {orig_n_test} Points")

def calculate_pretrained_embeddings(datasets: DatasetDict, tokenizer: T5TokenizerFast, pretrainer: PretrainedEmbeddingsModel):

    if isinstance(pretrainer, torch.nn.DataParallel):
        base_module = pretrainer.module
    else:
        base_module = pretrainer
    base_module.calculate_only_pretrained = True

    def embed_single_data(input_strings, name):
        input_tokenised = tokenizer(input_strings, truncation=False, padding=True, return_tensors='pt')
        input_tokenised = input_tokenised.to(next(iter(pretrainer.parameters())).device)
        with torch.no_grad():
            input_embeddings = pretrainer(input_tokenised.input_ids,
                                          input_tokenised.attention_mask,
                                          key_prefix=f'{name}_')
        input_embeddings = {k: v.cpu() if torch.is_tensor(v) else v for k, v in input_embeddings.items()}
        return input_embeddings

    def embed_data(input_batch):
        return {**embed_single_data(input_batch['chosen'], 'chosen'),
                **embed_single_data(input_batch['rejected'], 'rejected')}

    sample_batch = embed_data(datasets['train'][0].keys())
    sample_features = {k: get_float_feature_from_sample_array(v) for k, v in sample_batch.items()}
    sample_features = {k: v for k, v in sample_batch.items() if v is not None}
    sample_features = None if not sample_features else Features(map=sample_features)

    logger.info(f"Keys before Embedding: {datasets['train'][0].keys()}")
    datasets['train'] = datasets['train'].map(embed_data, batched=True, features=sample_features, with_indices=False, batch_size=768, load_from_cache_file=True)
    datasets['test'] = datasets['test'].map(embed_data, batched=True, features=sample_features, with_indices=False, batch_size=768, load_from_cache_file=True)
    datasets['val'] = datasets['val'].map(embed_data, batched=True, features=sample_features, with_indices=False, batch_size=768, load_from_cache_file=True)
    logger.info(f"Keys after Embedding: {datasets['train'][0].keys()}")

    base_module.calculate_only_pretrained = False


def prepare_dataset(data_params: DictConfig, tokenizer: T5TokenizerFast, data_rng: torch.Generator,
                    pretrainer: Optional[PretrainedEmbeddingsModel] = None):
    logger.info(f'Loading dataset with name {data_params.name} and data dir: {data_params.subdir}')
    with timer(f"Loading dataset {data_params.name}"):
        datasets: DatasetDict = load_dataset(data_params.name, data_dir=data_params.subdir)

    logger.info("Original Dataset stats:")
    logger.info(f"   Training Dataset  : {len(datasets['train'])} Points")
    logger.info(f"   Testing Dataset   : {len(datasets['test'])} Points")

    with timer(f"Filtering the datasets to contain points with length <= {data_params.max_tokens}"):
        filter_datasets_inplace_by_max_tokens_(datasets, tokenizer, data_params.max_tokens)

    train_dset: Dataset = datasets['train']
    test_dset: Dataset = datasets['test']

    train_val_test_dataset: DatasetDict = train_dset.train_test_split(test_size=data_params.val_fraction, seed=random_seed(data_rng).item())
    train_val_test_dataset['val'] = train_val_test_dataset['test']
    train_val_test_dataset['test'] = test_dset

    logger.info("Final Dataset stats:")
    logger.info(f"   Training Dataset  : {len(train_val_test_dataset['train'])} Points")
    logger.info(f"   Validation Dataset: {len(train_val_test_dataset['val'])} Points")
    logger.info(f"   Testing Dataset   : {len(train_val_test_dataset['test'])} Points")

    if pretrainer is not None:
        with timer(f"Calculating Embedding using model {pretrainer.__class__.__name__}"):
            calculate_pretrained_embeddings(train_val_test_dataset, tokenizer, pretrainer)

    # Assign device for use with my custom loaders
    for dset in train_val_test_dataset.values():
        dset.device = torch.device('cpu')

    return train_val_test_dataset
