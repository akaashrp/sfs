"""Run the frozen canonical control at the explicitly requested 8 QPS."""
import argparse

from scripts.cloud.canonical_control import execute


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=['preflight', 'run'])
    parser.add_argument('--bundle', required=True)
    parser.add_argument('--state', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--models')
    parser.add_argument('--gpus', choices=['0,1,2,3', '4,5,6,7'], required=True)
    options = parser.parse_args()
    options.qps = 8.0
    execute(options)
