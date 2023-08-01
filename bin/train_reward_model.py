import set_python_path
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

from training_loop import run_train_eval_loop
from training_loop import Checkpointer

from bin.train_reward_aux import RewardTracker
from bin.train_reward_aux import RewardProgressTracker
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
    use_multi_gpu = cfg.use_multi_gpu_if_available
    if not {'cuda' in cfg.device_name or 'gpu' in cfg.device_name} or torch.cuda.device_count() <= 1:
        use_multi_gpu = False

    torch.manual_seed(cfg.seed)
    data_rng = torch.Generator(torch.device('cpu'))  # data is generated on the CPu
    data_rng.manual_seed(data_params.seed)

    main_module: RewardMainModule = RewardMainModule(model_params=model_params, device=device, multi_gpu=use_multi_gpu)

    with timer("Loading and splitting the data (need to split val data additionally)"):
        if main_module.use_pretrained_output:
            pretrainer = main_module.reward_model
        else:
            pretrainer = None

        dataset: DatasetDict = prepare_dataset(data_params, main_module.tokenizer, data_rng, pretrainer=pretrainer)
        dataset['train'] = dataset['train'].with_format('torch', device=cfg.device_name)
        dataset['val'] = dataset['val'].with_format('torch', device=cfg.device_name)
        dataset['test'] = dataset['test'].with_format('torch', device=cfg.device_name)

    eff_n_epochs = train_params.loop.n_epochs + int(train_params.loop.n_batches > 0)
    optimizer, scheduler = init_optimizer(main_module, train_params.optimizer, eff_n_epochs)
    with timer("Running the Full Training Loop"):
        tensorboard_output_dir = opj(output_paths.simulation_path, 'tensorboard')
        checkpointer = Checkpointer(opj(output_paths.simulation_path, 'reward_model_cp.p'))
        main_module = run_train_eval_loop(dataset=dataset, 
                                          main_module=main_module, 
                                          optimizer=optimizer,
                                          loop_params=train_params.loop,
                                          data_rng=data_rng,
                                          train_tracker=RewardTracker(logger, 'Training', device),
                                          val_tracker=RewardTracker(logger, 'Validation', device, tensorboard_output_dir),
                                          test_tracker=RewardTracker(logger, 'Testing', device, tensorboard_output_dir),
                                          logger=logger,
                                          checkpointer=checkpointer,

                                          # optional
                                          scheduler=scheduler,
                                          progress_tracker=RewardProgressTracker(tensorboard_output_dir, logger, 'Training', device))

    with open(opj(output_paths.simulation_path, 'reward_model.p'), 'wb') as fout:
        dill.dump(main_module, fout, protocol=-1)

if __name__ == '__main__':
    # simmanager_context_decorator.debug = True
    main()