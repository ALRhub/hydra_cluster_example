# Hydra Cluster Example

This repository contains a simple example of a Hydra project and how to use it with a Slurm cluster. This Readme will serve as a tutorial for Hydra, WandB, and Slurm deployment.

## Installation
Create a new virtual environment (using conda, mamba, or virtualenv) and install the repository as a local package
(from the root of the repository):
```bash
pip install -e .
```
## Project Source Code
Have a look at the source files in the `src` directory. This projects implements a simple 1d regression task with a MLP model using torch and should
be fairly easy to understand. You notice that every class takes as an input a config dict. This is a common pattern in Hydra projects.

## Hydra Concept
Hydra is a powerful framework for configuring complex applications. It allows you to define  hyperparameter configurations 
and compose them in a hierarchical way. This is particularly useful when you want to combine experiments from multiple sub modules.

For example, you can define your algorithm and your dataset as separate modules and then combine them in a single experiment.
In this repository, we have 2 different datasets (Line or Sine 1d regression) and a simple MLP model as our algorithm.

The entry point of the project is the `main.py` file. In order to execute it, you have to provide a config file. You do this as a command line argument:
```bash
python main.py --config-name exp_1_local_sine
```
This executes the `main.py` script with the configuration `exp_1_local_sine` which is defined in the `configs` directory.
Taking a look at the config file, you can see that it first loads different subconfigs:
```yaml
defaults:
  - algorithm: mlp
  - dataset: sine
  - platform: local
  - _self_
```
After that, it sets specific paramters or overwrites default values:
```yaml

epochs: 1000
device: cuda
name: exp_1_local_sine
group_name: local
visualize: True
wandb: True
seed: 100
```
Importantly, in the `- platform: local` config, we define where the experiments should be saved. If you take a look at the config itself (`configs/platform/local.yaml`), you see that it starts with the line
```yaml
# @package _global_
```
This means, that it this config is "moved" to the global config level and that its keys are in the root level of the config instead inside the `platform` key.
We need that since the `hydra:` key needs to be in the root level of the config file.

## WandB Integration
This project is integrated with "Weights and Biases" (WandB). WandB is a great tool for tracking your experiments and visualizing your results. In order to use it, you have to create an account on the WandB website and get your API key.
Then, you can set your API key as an environment variable, WandB will tell you how to do this when you run it for the first time.

When you change the config `configs/exp_1_local_sine.yaml` and set the key `wandb: True` we will log our results to WandB.
Try it out by running this config again! You should see a new run in your WandB dashboard. However, the train loss curve is not very informative. 
You can change the y-axis to a logarithmic scale to see the loss better:
Hover over the train loss plot and click "Edit panel". Find the "Y Axis" option and tick the "Log Scale" button to the right.
Now you should see the loss curve better.

2 Things are recommended to change in the WandB GUI:
1. Group your runs to "Group" and "Job Type":
![img.png](data/grouping.png)
Groups are bigger gropus of runs, while Job Type can be seen as a sub group of a group. In our case, a job type only contains the same configuration, but we save multiple executions with different seeds in it.
2. Go to "Settings" (top right) -> "Line Plots" -> Tick "Random sampling" to see the correct grouping of multiple runs.
![img.png](data/random_sampling.png)


### Comparing Experiments on WandB
If you now start the second experiment `exp_2_local_relu` you will see that the runs are grouped by the "Group" and "Job Type" and that the line plots are correctly displayed.
Compare both runs. You should observe that ReLU converges faster than Tanh activation function (which was used by exp_1):

![img.png](data/relu_vs_tanh.png)

However, the ReLU prediction is a step wise linear function and looks janky, if you take a look a the the prediction plot:
![img.png](data/relu_vs_tanh_qualitative.png)

This is all easily compared in the WandB GUI.

## Hydra Sweeper and Multirun
Hydra also provides a powerful tool for running multiple experiments in parallel. This is done with the `MULTIRUN` flag.

Take a look at `configs/exp_3_local_multirun.yaml`. It has this new section:

```yaml
hydra:
  mode: MULTIRUN
  sweeper:
    params:
      seed: 0, 1, 2  # starts 3 jobs sequentially, overwrites seed value
```
With this configuration, we start 3 jobs sequentially with different seeds. This is useful if you want to increase the statistical significance of your results.
Also, all runs on the cluster are only executed using the MULTIRUN mode.

In WandB, you should see now an aggregated line plot of these 3 runs, when selecting the group `local_multirun` to visualize.
![img.png](data/multi_run_result.png)

Specific aggregation methods can be changed by the "Edit panel" option.

The `sweeper` section can be used to define a grid search: It creates a cartesian grid of all combinations given in the params dict.
If you are interested in a "list"/"tuple" search, check out my repository
[hydra-list-sweeper](https://github.com/ALRhub/hydra_list_sweeper).

## Slurm Cluster

Now we are ready to deploy our code on a Slurm cluster.

