"""Isolated subset/full Qwen FCFS pool (the `fcfs` profile of scripts.cloud.serving.pool)."""
from scripts.cloud.serving import pool as generic
from scripts.cloud.serving.profiles import FCFS


def pool(bundle, models, output, gpus, indices=(0,1,2), coefficients=None):
    return generic.pool(FCFS,bundle,models,output,gpus,indices,coefficients)
