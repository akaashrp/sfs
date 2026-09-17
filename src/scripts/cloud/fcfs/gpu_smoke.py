"""Bounded startup/memory/nonchunking evidence for the FCFS profile (scripts.cloud.serving.gpu_smoke --profile fcfs)."""
import argparse
from pathlib import Path

from scripts.cloud.common import read
from scripts.cloud.serving.gpu_smoke import run
from scripts.cloud.serving.profiles import FCFS


def main():
    p=argparse.ArgumentParser();p.add_argument('--bundle',type=Path,required=True);p.add_argument('--models',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--gpus',required=True);p.add_argument('--indices',default='0,1,2');a=p.parse_args()
    run(FCFS,a.bundle,read(a.models),a.output,a.gpus.split(','),tuple(map(int,a.indices.split(','))))

if __name__=='__main__':main()
