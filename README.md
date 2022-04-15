# Plasticity and Evolvability under Environmental Variability

This is the codebase for our paper "Plasticity and evolvability under environmental variability: the joint role of fitness-based selection and niche-limited competition" presented at GECCO 2022. 

In this work we have studied the evolution of a population of agents in a world where the fitness landscape changes with generations based on climate function and a latitudinal model that divides the world in different niches. We have implemented different selection mechanisms (fitness-based selection and niche-limited competition) and introduced many evaluation metrics to characterize the phenotypic and genotypic diversity of the population.

The repo contains the following main elements :

* folder source contains the main functionality for running a simulation
* scripts/run/reproduce_gecco.py can be used to rerun all simulations in the paper
* scripts/evaluate contains scripts for reproducing figures. reproduce_figures.py will produce all figures (provided you have already run scripts/run/reproduce_gecco.py to generate the data)
* folder projects contains data generated from running a simulation

# How to run

To install all package dependencies you can create a conda environment as:

`conda env create -f environment.yml`

All script executions need to be run from folder source. Once there, you can use simulate.py, the main interface of the codebase to run a simulation, For example:

`python simulate.py --project test_stable --env_type stable --num_gens 300 --capacity 1000 --num_niches 10 --trials 10 --selection_type NF --climate_mean_init 2`

will run a simulation with an environment with a climate function whose state is constantly 2 consisting of 100 niches for 300 generations and 10 independent trials. The maximum population size will be 1000*2 and selection will be fitness-based (higher fitness means higher chances of reproduction) and niche limited (individuals reproduce independently in each niche and compete only within a niche),

You can also take a look at scripts/run/reproduce_gecco.py to see which flags were used for the simulations presented in the paper.

