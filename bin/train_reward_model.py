from os.path import join as opj
import logging

import torch

from simmanager import Paths
from simmanager.tools import Timer
from datasets import DatasetDict

from rlhf_flant5.utils.basicutils import simmanager_context_decorator
from rlhf_flant5.utils.basicutils import loggingext_decorator
from rlhf_flant5.utils.basicutils import getFrameDir
from rlhf_flant5.utils.hydrashim import hydra, DictConfig
from rlhf_flant5.utils.dillshim import dill

from bin.training_loop import run_train_eval_loop

from bin.train_reward_aux import RewardTracker
from bin.train_reward_aux import RewardMainModule
from bin.train_reward_aux import prepare_dataset
from bin.train_reward_aux import init_optimizer

logger = logging.getLogger('train_reward_model')
timer = Timer(logger, 'INFO')

script_dir = getFrameDir()

@hydra.main(config_path=f'{script_dir}/../config', config_name='reward_model_config')
@simmanager_context_decorator
@loggingext_decorator
def main(cfg: DictConfig, output_paths: Paths):

    train_params: DictConfig = cfg.training
    model_params: DictConfig = cfg.model
    data_params: DictConfig = cfg.data

    device = torch.device(cfg.device_name)

    torch.manual_seed(cfg.seed)
    data_rng = torch.Generator(torch.device('cpu'))  # data is generated on the CPu
    data_rng.manual_seed(data_params.seed)

    main_module: RewardMainModule = RewardMainModule(model_params=model_params).to(device)

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
    import ipdb; ipdb.set_trace()
    simmanager_context_decorator.debug = True
    main()