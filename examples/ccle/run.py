import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import os
import argparse
import torch
from main import load_plx4720, run_hrt, CvModel, get_model_weights
from multiprocessing import Pool
from pyhrt.utils import bh


def run_parallel(x):
    feat_idx, X_drug, y_drug, elastic_model, features, ccle_features, model_weights = x
    p_path = 'data/p_{}.npy'.format(feat_idx)

    # Check if we have already computed this p-value
    if os.path.exists(p_path):
        p_value = np.load(p_path)
        print('p-value for {}: {}'.format(ccle_features[feat_idx], p_value))
        return p_value
    
    # Check if the model assigns zero weight to this feature such that it can be ignored
    if model_weights[feat_idx] == 0:
        print('p-value for {}: 1 (0 weight in model)'.format(ccle_features[feat_idx]))
        return 1
        
    # If the p-value hasn't been computed yet, generate it as save it 
    p_value = run_hrt(feat_idx, X_drug, y_drug, elastic_model, features, ccle_features)
    print('p-value for {}: {}'.format(ccle_features[feat_idx], p_value))
    np.save(p_path, p_value)
    return p_value

def seed_fn():
    np.random.seed(42)
    torch.manual_seed(42)
    torch.set_num_threads(1)

def main():
    parser = argparse.ArgumentParser(description='Worker script for the case study.')

    # Experiment settings
    parser.add_argument('min_feature', type=int, help='Min Feature ID')
    parser.add_argument('max_feature', type=int, help='Max Feature ID. Set to -1 to go to the max feature ID.')
    parser.add_argument('--nthreads', type=int, default=4, help='Number of parallel workers to run.')
    parser.add_argument('--fdr_threshold', type=float, default=0.2, help='Target false discovery rate.')

    # Get the arguments from the command line
    args = parser.parse_args()
    dargs = vars(args)

    torch.set_num_threads(1) # bad torch, no biscuit

    # Load the data and the model
    print('Loading data')
    X_drug, y_drug, features, ccle_features, elastic_model = load_plx4720()

    if args.max_feature == -1:
        args.max_feature = len(ccle_features)

    # Get the weights to use in feature ranking
    model_weights = get_model_weights(elastic_model)

    # Build all the jobs
    jobs = [(feat_idx, X_drug, y_drug, elastic_model, features, ccle_features, model_weights) for feat_idx in range(args.min_feature, min(args.max_feature+1,len(ccle_features)))]

    print('Running {} jobs'.format(len(jobs)))

    # Run in parallel (okay this is processes not threads, but who's counting?)
    with Pool(args.nthreads, initializer=seed_fn) as pool:
        p_values = np.array(pool.map(run_parallel, jobs))

    



if __name__ == '__main__':
    main()


