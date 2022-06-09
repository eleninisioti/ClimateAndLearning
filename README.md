# Plasticity and Evolvability under Environmental Variability

This is the code accompannying our paper [Plasticity and evolvability under environmental variability: the joint role of fitness-based selection and niche-limited competition" ](https://arxiv.org/abs/2202.08834) which is to be presented at the Gecco 2022 conference.

In this work we have studied the evolution of a population of agents in a world where the fitness landscape changes with generations based on climate function and a latitudinal model that divides the world in different niches. We have implemented different selection mechanisms (fitness-based selection and niche-limited competition).

The world is divided into niches that correspond to different latitudes and whose state evolves based on a common climate function:

![World model](https://firebasestorage.googleapis.com/v0/b/firescript-577a2.appspot.com/o/imgs%2Fapp%2Feleni%2FJRz9LWAvHU.png?alt=media&token=62d91979-1732-4014-8bef-50858dac979c)

We model the plasticity of an individual using tolerance curves  originally developed in ecology. Plasticity curves have the form of a Gaussian the capture the benefits and costs of plasticity when comparing a specialist (left) with a generalist (right) agent:



![tolerance curve](https://firebasestorage.googleapis.com/v0/b/firescript-577a2.appspot.com/o/imgs%2Fapp%2Feleni%2Fog3I7VZYFz.png?alt=media&token=af97defa-4ccc-46e4-bf50-3eb581ff9cf1)

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

Running all simulations requires some days. You can instead download the data produced by running scripts/run/reproduce_gecco.py from [this google folder](https://drive.google.com/file/d/1rwyEx7n7mqJuy5LHkbpn0w3sxBYE5J6t/view?usp=sharing) and unzip them under the projects directory. 

