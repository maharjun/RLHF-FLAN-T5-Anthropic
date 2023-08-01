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
    model.train_transformer=False \
    model.use_pretrained_output=False \
    training.loop.train_batch_size=256 \
    training.loop.val_batch_size=1024 \
    training.loop.n_epochs=6
    index=1
```

Since we're using hydra for the configuration, any of the parameters found in reward_model_config can be changed on the command line. You may change the `desc` (description) and `index` parameter to create a unique directory to store the results

The logs, tensorboard data, and the best model by validation error can then be accessed in the directory `$RESULTS_ROOT_DIR/Reward-Model-Training/desc-<description>,index-<index>`.

# File Structure

1. hydra config: config/reward_model_config.py
2. main file: bin/train_reward_model.py
3. models in models/reward_models