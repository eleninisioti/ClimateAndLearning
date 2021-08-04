# ClimateAndLearning



## Repository structure

Source files are under the directory `source`. Scripts for plotting and running experiments are under `scripts`. `Material` contains various documents related to the project and results from existing experiments are under `projects`.

The existing experiments reproduce some results from "Evolution and dispersal under climatic instability: a simple evolutionary algorithm" under `Grove_2014` and extend these results to Maslin's climate model under `Maslin`.

## How to run

Install required dependencies by running

`conda env create -f environment.yml`

File `simulate.py` is the main interface and accepts a variety of arguments. It needs to be called from the `source` directory. For example, a simulation on Maslin's climate model can be run as:

`python simulate.py --project Maslin/my_project --env_type combined --model hybrid --num_gens 2500`

This creates a new directory under `projects` which contains a `config.yml` file for storing the flags and reproducing the experiment, a `log.pickle` with all collected data used to generate plots and a `plots` directory.

## Modeling

The code allows for experimenting with multiple models/design choices.

The options for the climate models are:

* a simple pulse, chosen with `--env_type change`
* a sinusoid, chosen with `--env_type sin`
* Maslin's model, chosen with `--env_type combined`

The options for the evolutionary models are:

* A. Identical to Grove's model.
* hybrid. This model has a carrying capacity and removes extinct individuals

* hybrid_nocapac. There is no carrying capacity and removes extinct individuals
* hybrid_noextinct. There is a carrying capacity and does not remove extinct individuals
* MC. Instead of survival of the fittest, a minimum criterion is used for determining which individuals reproduce



