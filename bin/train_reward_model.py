from os.path import join as opj
import logging
from dataclasses import dataclass

import torch

from simmanager import Paths
from simmanager.tools import Timer
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, AutoConfig
from transformers.models.t5.modeling_t5 import T5ForConditionalGeneration, T5Stack
from transformers.models.t5.tokenization_t5_fast import T5TokenizerFast
from transformers.models.t5.configuration_t5 import T5Config

from rlhf_flant5.utils.basicutils import simmanager_context_decorator
from rlhf_flant5.utils.basicutils import loggingext_decorator
from rlhf_flant5.utils.basicutils import getFrameDir
from rlhf_flant5.utils.hydrashim import hydra, DictConfig
from rlhf_flant5.utils.dillshim import dill

# from rlhf_flant5.utils.data.loader import random_batch_dataset
from rlhf_flant5.utils.torchutils import random_seed
from rlhf_flant5.models.rewardmodels import RewardFromLayerwiseWeightedAttention

from bin.training_loop import MainModule
from bin.training_loop import ValidationTracker
from bin.training_loop import run_train_eval_loop

from torchmetrics.aggregation import MeanMetric
from torchmetrics.classification import BinaryAccuracy

from datasets import load_dataset, DatasetDict, Dataset

logger = logging.getLogger('train_reward_model')
timer = Timer(logger, 'INFO')

script_dir = getFrameDir()

@dataclass
class RewardMainOutput:
    chosen_input_tokenised: torch.Tensor
    rejected_input_tokenised: torch.Tensor
    chosen_rewards: torch.Tensor
    rejected_rewards: torch.Tensor
    loss: torch.Tensor


class RewardMainModule(MainModule):
    Output = RewardMainOutput

    def __init__(self, model_params: DictConfig):
        super().__init__()

        pretrained_transformer: T5ForConditionalGeneration = \
            AutoModelForSeq2SeqLM.from_pretrained(model_params.transformer_name) 

        # Disable training for pretrained_encoder
        pretrained_encoder: T5Stack = pretrained_transformer.encoder
        for parameter in pretrained_encoder.parameters():
            parameter.requires_grad = False


        self.tokenizer: T5TokenizerFast = AutoTokenizer.from_pretrained(model_params.transformer_name)
        self.transformer_config: T5Config = AutoConfig.from_pretrained(model_params.transformer_name)

        self.reward_model = RewardFromLayerwiseWeightedAttention(pretrained_encoder,
                                                                 model_params.attention_inner_dim,
                                                                 model_params.pooling_output_dim,
                                                                 model_params.readout_additional_layers)

    def device(self):
        return self.reward_model.cross_layer_weighted_pooling.W.device

    def output(self, batch_dataset_struct):
        chosen_input = batch_dataset_struct['chosen']
        rejected_input = batch_dataset_struct['rejected']

        chosen_input_tokenised = self.tokenizer(chosen_input, truncation=True, padding=True, return_tensors='pt')  # max_len = 512
        rejected_input_tokenised = self.tokenizer(rejected_input, truncation=True, padding=True, return_tensors='pt')  # max_len = 512

        assert len(chosen_input_tokenised.input_ids) == len(rejected_input_tokenised.input_ids) == len(chosen_input) == len(rejected_input)
        chosen_rewards = self.reward_model(chosen_input_tokenised)
        rejected_rewards = self.reward_model(rejected_input_tokenised)

        loss = -torch.nn.functional.logsigmoid(chosen_rewards - rejected_rewards)

        return RewardMainOutput(chosen_input_tokenised=chosen_input_tokenised,
                                rejected_input_tokenised=rejected_input_tokenised,
                                chosen_rewards=chosen_rewards,
                                rejected_rewards=rejected_rewards,
                                loss=loss)

    def backward(self, module_output: RewardMainOutput):
        total_loss = torch.sum(module_output.loss, dim=0)
        assert total_loss.ndim == 0
        total_loss.backward()


class RewardTracker(ValidationTracker):
    def __init__(self, logger: logging.Logger, stage_name: str, device: torch.device):
        self.logger = logger
        self.stage_name = stage_name

        self.mean_loss = MeanMetric().to(device=device)
        self.mean_accuracy = BinaryAccuracy().to(device=device)

    def update(self, module_output: RewardMainOutput):
        self.mean_loss.update(module_output.loss)
        self.mean_accuracy.update(torch.sigmoid(module_output.chosen_rewards - module_output.rejected_rewards),
                                  torch.ones_like(module_output.chosen_rewards))

    def reset(self):
        self.mean_loss.reset()
        self.mean_accuracy.reset()

    def validation_loss(self):
        return self.mean_loss.compute()

    def log(self):
        self.logger.info(f"Mean {self.stage_name} LogSigmoid Loss: {self.mean_loss.compute():.6}")
        self.logger.info(f"Mean {self.stage_name} Accuracy: {self.mean_accuracy.compute():.6}")


def init_optimizer(main_module: MainModule, optimizer_params: DictConfig):
    if optimizer_params.type == 'AdamW':
        optimizer = torch.optim.AdamW(main_module.parameters(), lr=optimizer_params.lr, weight_decay=optimizer_params.weight_decay)
    elif optimizer_params.type == 'Adam':
        optimizer = torch.optim.Adam(main_module.parameters(), lr=optimizer_params.lr, weight_decay=optimizer_params.weight_decay)
    else:
        raise ValueError("The training.optimizer.type should be one of {'AdamW', or 'Adam'}")
    return optimizer


def prepare_dataset(data_params: DictConfig, data_rng: torch.Generator):
    logger.info(f'Loading dataset with name {data_params.name} and data dir: {data_params.subdir}')
    dataset: DatasetDict = load_dataset(data_params.name, data_dir=data_params.subdir)

    train_dset: Dataset = dataset['train']
    test_dset: Dataset = dataset['test']

    train_val_test_dataset: DatasetDict = train_dset.train_test_split(test_size=data_params.val_fraction, seed=random_seed(data_rng).item())
    train_val_test_dataset['val'] = train_val_test_dataset['test']
    train_val_test_dataset['test'] = test_dset

    # Assign device for use with my custom loaders
    for dset in train_val_test_dataset.values():
        dset.device = torch.device('cpu')

    return train_val_test_dataset


@hydra.main(config_path=f'{script_dir}/../config', config_name='reward_model_config')
@simmanager_context_decorator
@loggingext_decorator
def main(cfg: DictConfig, output_paths: Paths):

    train_params: DictConfig = cfg.training
    model_params: DictConfig = cfg.model
    data_params: DictConfig = cfg.data

    device = torch.device(cfg.device_name)

    torch.manual_seed(cfg.seed)
    data_rng = torch.Generator(device)
    data_rng.manual_seed(data_params.seed)

    main_module: MainModule = RewardMainModule(model_params=model_params)

    with timer("Loading and splitting the data (need to split val data additionally)"):
        dataset: DatasetDict = prepare_dataset(data_params, data_rng)

    optimizer = init_optimizer(main_module, train_params.optimizer)

    with timer("Running the Full Training Loop"):
        main_module = run_train_eval_loop(dataset=dataset, 
                                          main_module=main_module, 
                                          optimizer=optimizer,
                                          loop_params=train_params.loop,
                                          data_rng=data_rng,
                                          train_tracker=RewardTracker(logger, 'Training', device),
                                          val_tracker=RewardTracker(logger, 'Validation', device),
                                          test_tracker=RewardTracker(logger, 'Testing', device),
                                          logger=logger)

    with open(opj(output_paths.simulation_path, 'reward_model.p'), 'wb') as fout:
        dill.dump(main_module, fout, protocol=-1)

if __name__ == '__main__':
    # import ipdb; ipdb.set_trace()
    simmanager_context_decorator.debug = True
    main()