RLHF with FLAN-T5 and Anthropic dataset
=======================================

Creating the environment
------------------------

The environment is a combination of conda and pip packages. It is advisable to install the pip packages in a virtual env within the conda environment as there are some dependency issues with certain libraries.

1. **Install conda dependencies**: In order to do this, change to the repository directory and run the following command

    ```
    conda create --yes -f environment.yml
    ```

2. **Create venv for pip dependencies**: Now choose a directory OUTSIDE the repository (here `../rlhf-venv`) for the virtual environment and run the following to create the environment

    ```
    conda activate rlhf-flant5-env
    python -m venv --system-site-packages ../rlhf-venv
    ```

3. **Install pip requirements**:

    ```
    source ../rlhf-venv/bin/activate
    pip install -r requirements.txt
    ```

3. **Set environment variables**

    * `RESULTS_ROOT_DIR`: This should point to an *existing* directory outside the
      repository that will contain all the results of the simulation. Note: The code
      makes use of the SimManager package (https://github.com/IGITUGraz/SimManager) to
      ensure reproducibility, which requires this variable be set
    * (Optional) `HF_HOME`: This should point to a directory that will hold the cached datasets and models from huggingface

When running the code, you may source the following script to setup the environment once installed

```
micromamba activate rlhf-flant5-env
source <absolute-path-to>/rlhf-venv/bin/activate
export RESULTS_ROOT_DIR=<results-directory>
```

How to run the code
-------------------

### Training The Reward model

The configuration for training the reward model is in the file `config/reward_model_config.yaml`. To train the reward model, run the following command.


```
python bin/train_reward_model.py desc=ete_attention_pooled_encoder \
    model=attention_pooled_encoder \
    model.train_transformer=True \
    model.use_pretrained_output=False \
    use_multi_gpu_if_available=True \
    training.loop.train_batch_size=96 \
    training.loop.val_batch_size=1024 \
    training.loop.n_epochs=6
    index=1
```

Since we're using hydra for the configuration, any of the parameters found in reward_model_config can be changed on the command line. You may change the `desc` (description) and `index` parameter to create a unique directory to store the results. Additionally, Change the batch sizes to fit GPU memory. The above is run using 4 GPU's of 16GB each.

The logs, tensorboard data, and the best model by validation error can then be accessed in the directory `$RESULTS_ROOT_DIR/Reward-Model-Training/desc-<description>,index-<index>`. In this case it would be the directory `$RESULTS_ROOT_DIR/Reward-Model-Training/desc-ete_attention_pooled_encoder,index-1`

> NOTE: SimManager will refuse to run the simulation if:
> 
> 1. There are any untracked files in the repository that are not ignored by .gitignore
> 2. The destination directory already exists (e.g. created in a previous call). In
>    that case, simply change the description or update the index to create a new
>    directory

File Structure
--------------

### config


The hydra configs for reward modeling and the various transformer readout models tried are here.

1. `reward_model_config.yaml`: Configuration for reward modeling
2. `model/*`: The configurations for each of the different transformer readout models

### bin directory

All the code corresponding to execution of the training loop

1. `bin/train_reward_model.py`: The main script to run the training loop 
2. `bin/train_reward_aux.py`: Auxiliary utilities pertaining to reading the config, and
   initializing the necessary loop modules, optimizers, and data.
3. `bin/training_loop.py`: Implementation of a generic feature-rich, configurable training loop. 

### rlhf_flant5/models

The directory containing the implementation of all the various transformer readout models.

#### Models implemented:

All the following models are implemented in the file `rewardmodels.py`

1. *RewardFromLayerwiseAttention*: the hidden_states in each layer
   of the encoder are pooled via attention pooling and combined across layers via
   weighted pooling

2. *RewardFromFullAttentionPooling*: the hidden_states in each layer
   of the encoder are pooled via attention pooling and combined across layers via
   attention pooling

3. *RewardFromDecoderOutput*: the final output of the decoder is
   used to readout the reward. In this model, the decoder output can be precomputed
   for when the transformer is not trained.

4. *RewardFromAttentionPooledEncoder*: the state corresponding to the first token
   in each layer of the encoder is pooled together via attention pooling. In this
   model, the encoder hidden states can be precomputed for when the transformer is
   not trained.

4. *RewardFromAttentionPooledDecoder*: the state corresponding to the first token
   in each layer of the decoder is pooled together via attention pooling. In this
   model, the decoder hidden states can be precomputed for when the transformer is
   not trained.

### rlhf_flant5/utils directory

Generic Utilities directory 

> Code in this directory is code I have written previously and not exclusively for this task