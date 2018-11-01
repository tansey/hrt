import argparse
import numpy as np
import torch
from multiprocessing import Pool
from sim_liang import run

def run_parallel(x):
    trial = x[0]
    feature = x[1]
    reset = x[2]

    torch.set_num_threads(1) # bad torch, no biscuit

    for cv in [False, True]:
        for robust in [False, True]:
            run(trial, feature, reset, cv, robust)

def seed_fn():
    np.random.seed()

def main():
    parser = argparse.ArgumentParser(description='Worker script for the benchmark.')

    # Experiment settings
    parser.add_argument('min_trial', type=int, help='Min Trial ID')
    parser.add_argument('max_trial', type=int, help='Max Trial ID')
    parser.add_argument('min_feature', type=int, help='Min Feature ID')
    parser.add_argument('max_feature', type=int, help='Max Feature ID')
    parser.add_argument('--reset', action='store_true', help='If specified, resets the models.')
    parser.add_argument('--nthreads', type=int, default=8, help='Number of parallel workers to run.')

    # Get the arguments from the command line
    args = parser.parse_args()
    dargs = vars(args)

    torch.set_num_threads(1) # bad torch, no biscuit

    # Build all the jobs
    jobs = []
    for f in range(args.min_feature, args.max_feature+1):
        for t in range(args.min_trial, args.max_trial+1):
            jobs.append((t,f,args.reset))

    # Run in parallel (okay this is processes not threads, but who's counting?)
    with Pool(args.nthreads, initializer=seed_fn) as p:
        p.map(run_parallel, jobs)


if __name__ == '__main__':
    main()



