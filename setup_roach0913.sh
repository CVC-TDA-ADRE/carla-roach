#!/bin/sh

# For getting data
# TDA_Latest
export COIL_DATASET_PATH=~/datasets/CARLA/
export COIL_SYNTHETIC_DATASET_PATH=~/datasets/CARLA/
export COIL_REAL_DATASET_PATH=~/datasets/CARLA/
#export CARLA_ROOT=/home/dporres/Documents/TDA_Latest
export CARLA_ROOT=~/carla
#export CARLA_ROOT=/home/dporres/Documents/TDA_v13_1/
export PYTHONPATH=${CARLA_ROOT}/PythonAPI/carla/dist/carla-0.9.13-py3.7-linux-x86_64.egg
export PYTHONPATH=${CARLA_ROOT}/PythonAPI/carla/:${PYTHONPATH}

export PYTHONPATH=$(pwd):${PYTHONPATH}
export PYTHONPATH=~/TDA/carla-roach/agents/:${PYTHONPATH}
