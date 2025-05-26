<h1>  Agent-Arena (v0): A Simple Framework for Experimenting Control Algorithms on Benchmark Environments </h1>

This project is mainly authored by **Halid Abdulrahim Kadi** and supervised by **Dr Kasim Terzić** at the *Univiersity of St Andrews, UK*.

**Ryan Hayward** partially contributed to the `README.md` of the forked `softgym` repository that works align with this framework.

**Jose Alex Chandy** participated in the development of `environment.yml` file in the `ros1-integration` repository.


[![DOI](https://zenodo.org/badge/933415395.svg)](https://doi.org/10.5281/zenodo.14876793)

[ArXiv](https://arxiv.org/abs/2504.06468)

![plot](assets/agent-arena.jpg)


<h3>  Branch to fit ROS1 Noetic</h3>

This branch is for integrating the `Agent-Arena` framework with `ROS1 Noetic`. The only difference between this branch and main is in its `environment.yml` file, where it also includes the necessary packages to make Agent-Arena work with `ROS1 Noetic`. On the robot machine, please follow the `README.md` in the `main` branch to install `agent-arena-v0`.


##  Setup `ROS` with `agent-arena`
```
cd <path-to-agent-arena>#

# Establish agent-arena conda environment
. ./setup.sh  

# Integarate ROS1 Noetic enviornment.
source $CONDA_PREFIX/setup.bash 
```