import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import os
import argparse
import torch
from main import load_olfaction, load_or_fit_model, run_hrt, CvModel, get_model_weights
from multiprocessing import Pool
from pyhrt.utils import bh


def run_parallel(x):
    target_feature, x, y, features, forest_model, model_weights, descriptor = x
    feat_idx = features.get_loc(target_feature)
    p_path = 'data/p_{}_{}.npy'.format(descriptor, feat_idx)

    # Check if we have already computed this p-value
    if os.path.exists(p_path):
        p_value = np.load(p_path)
        print('p-value for {}: {}'.format(target_feature, p_value))
        return p_value
    
    # Check if the model assigns zero weight to this feature such that it can be ignored
    if model_weights[feat_idx] == 0:
        print('p-value for {}: 1 (0 weight in model)'.format(target_feature))
        return 1
        
    # If the p-value hasn't been computed yet, generate it as save it 
    p_value = run_hrt(target_feature, x, y, features, forest_model)
    print('p-value for {}: {}'.format(target_feature, p_value))
    np.save(p_path, p_value)
    return p_value

def seed_fn():
    np.random.seed(42)
    torch.manual_seed(42)
    torch.set_num_threads(1)

def main():
    parser = argparse.ArgumentParser(description='Worker script for the case study.')

    # Experiment settings
    descriptors = ['Bakery', 'Sour','Intensity','Sweet','Burnt','Pleasantness','Fish', 'Fruit',
                   'Garlic','Spices','Cold','Acid','Warm',
                   'Musky','Sweaty','Ammonia','Decayed','Wood','Grass',
                   'Flower','Chemical']
    parser.add_argument('min_feature', type=int, help='Min Feature ID')
    parser.add_argument('max_feature', type=int, help='Max Feature ID')
    parser.add_argument('--descriptor', choices=descriptors, default='Bakery', help='The descriptor type to get p-values for.')
    parser.add_argument('--nthreads', type=int, default=4, help='Number of parallel workers to run.')
    parser.add_argument('--fdr_threshold', type=float, default=0.2, help='Target false discovery rate.')
    parser.add_argument('--importance_threshold', type=float, default=1e-3, help='Minimum heuristic feature importance to make a feature test-worthy.')

    # Get the arguments from the command line
    args = parser.parse_args()
    dargs = vars(args)

    torch.set_num_threads(1) # bad torch, no biscuit

    # Load the data and the model
    print('Loading data')
    X, Y, descriptors, target_features = load_olfaction()
    features = X.columns

    print('Loading model')
    # Get the model and data specifically for this descriptor class
    x, y, forest_model = load_or_fit_model(args.descriptor, X, Y)

    # Handle an idiosyncracy of multiprocessing with sklearn random forests
    for m in forest_model.models:
        m.n_jobs = 1

    # Get the weights to use in feature ranking
    model_weights = get_model_weights(forest_model)

    print('Total: {}'.format(len(X.columns)))

    # Build all the jobs
    jobs = [(target_feature, x, y, features, forest_model, model_weights, args.descriptor)
                for target_feature, importance in zip(X.columns[args.min_feature:args.max_feature+1],
                                                      model_weights[args.min_feature:args.max_feature+1])
                if importance >= args.importance_threshold]

    print('Running {} jobs'.format(len(jobs)))

    # Run in parallel (okay this is processes not threads, but who's counting?)
    with Pool(args.nthreads, initializer=seed_fn) as pool:
        p_values = np.array(pool.map(run_parallel, jobs))

    # Multiple testing correction via Benjamini-Hochberg at a 20% FDR
    discoveries = bh(p_values, args.fdr_threshold)
    discovery_genes = features[discoveries]
    discovery_p = p_values[discoveries]
    discovery_weights = model_weights[discoveries]

    # Print the discoveries along with their model weights and p-values
    order = np.argsort(discovery_weights)[::-1]
    print('')
    print('Molecular Feature & Model Weight & $p$-value \\\\')
    for g,w,p in zip(discovery_genes[order],
                     discovery_weights[order],
                     discovery_p[order]):
        print('{} & {} & {} \\\\'.format(g,w,p))



if __name__ == '__main__':
    main()


