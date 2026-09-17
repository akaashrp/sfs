"""Refit and audit SFS batch-latency coefficients from destination FCFS traces (the `fcfs` profile of scripts.cloud.serving.coefficients)."""
import argparse
from pathlib import Path

from scripts.cloud.serving import coefficients as generic
from scripts.cloud.serving.coefficients import NAMES, FEATURES, MINIMUM_R2, traces, predict, diagnostics
from scripts.cloud.serving.profiles import FCFS


def fit(calibration, output):return generic.fit(calibration,output,FCFS)


def load(path):return generic.load(path,FCFS)


def validate(coefficients_path, qualification, output):return generic.validate(coefficients_path,qualification,output,FCFS)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('mode',choices=['fit','validate'])
    p.add_argument('--calibration',type=Path,help='worker output directory holding calibration_trace_<model>.csv')
    p.add_argument('--coefficients',type=Path);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if a.mode=='fit':fit(a.calibration,a.output)
    else:validate(a.coefficients,a.calibration,a.output)
    print(a.output)
