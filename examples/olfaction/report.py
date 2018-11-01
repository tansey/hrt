import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import os
import argparse
import torch
from main import load_olfaction, load_or_fit_model, CvModel, get_model_weights
from pyhrt.utils import bh

def main():
    parser = argparse.ArgumentParser(description='Worker script for the case study.')

    # Experiment settings
    descriptors = ['Bakery', 'Sour','Intensity','Sweet','Burnt','Pleasantness','Fish', 'Fruit',
                   'Garlic','Spices','Cold','Acid','Warm',
                   'Musky','Sweaty','Ammonia','Decayed','Wood','Grass',
                   'Flower','Chemical']
    parser.add_argument('--descriptor', choices=descriptors, default='Bakery', help='The descriptor type to get p-values for.')
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

    all_p_path = 'data/p_{}.npy'.format(args.descriptor)
    if os.path.exists(all_p_path):
        p_values = np.load(all_p_path)
    else:
        p_values = np.full(len(features), np.nan)
    for feat_idx in range(len(features)):
        if not np.isnan(p_values[feat_idx]):
            continue

        p_path = 'data/p_{}_{}.npy'.format(args.descriptor, feat_idx)

        # Check if we have already computed this p-value
        if os.path.exists(p_path):
            p_values[feat_idx] = np.load(p_path)
            # print('p-value for {}: {}'.format(features[feat_idx], p_values[feat_idx]))
            continue
        
        # Check if the model assigns zero weight to this feature such that it can be ignored
        if np.abs(model_weights[feat_idx]) < args.importance_threshold:
            # print('p-value for {}: 1 (0 weight in model)'.format(features[feat_idx]))
            p_values[feat_idx] = 1
            continue
    
        print('************ Missing p-value for {} ************'.format(feat_idx))

    # Save the aggregated results
    np.save(all_p_path, p_values)

    # Print the top-ranked features by their heuristic weight
    for rank, (target_feature, importance) in enumerate(target_features[args.descriptor]):
        p_value = p_values[features.get_loc(target_feature)]
        print('{} & {:.4f} & {:.4f} \\\\'.format(target_feature.replace('\'',''), importance, p_value))

    if np.any(np.isnan(p_values)):
        print('{} NaN p-values!'.format(np.isnan(p_values).sum()))
        missing = np.where(np.isnan(p_values))[0]
        print(missing)
        print([features[idx] for idx in missing])
        print('Setting to 1')
        p_values[np.isnan(p_values)] = 1

    # Only consider features with substantial heuristic feature importance
    important = model_weights >= args.importance_threshold
    p_important = p_values[important]

    print('{} features above importance threshold'.format(important.sum()))
    
    # Multiple testing correction via Benjamini-Hochberg at a 20% FDR
    # discoveries = bh(p_values, args.fdr_threshold) # Test all features
    discoveries = np.arange(len(features))[important][bh(p_important, args.fdr_threshold)] # Test important features
    discovery_features = features[discoveries]
    discovery_p = p_values[discoveries]
    discovery_weights = model_weights[discoveries]

    # Print the discoveries along with their model weights and p-values
    order = np.argsort(np.abs(discovery_weights))[::-1]
    print('')
    print('Molecular Feature & Model Weight & $p$-value \\\\')
    for f,w,p in zip(discovery_features[order],
                     discovery_weights[order],
                     discovery_p[order]):
        print('{} & {:.4f} & {:.4f} \\\\'.format(f,w,p))



if __name__ == '__main__':
    main()
    


