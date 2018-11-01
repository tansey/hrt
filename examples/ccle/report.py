import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import os
import argparse
import torch
from main import load_plx4720, run_hrt, CvModel, get_model_weights
from pyhrt.utils import bh

if __name__ == '__main__':
    fdr_threshold = 0.2
    importance_threshold = 1e-3

    # Load the data and the model
    print('Loading data')
    X_drug, y_drug, features, ccle_features, elastic_model = load_plx4720()

    # Get the weights to use in feature ranking
    model_weights = get_model_weights(elastic_model)

    all_p_path = 'data/p.npy'
    if os.path.exists(all_p_path):
        p_values = np.load(all_p_path)
    else:
        p_values = np.full(len(ccle_features), np.nan)
    for feat_idx in range(len(ccle_features)):
        if not np.isnan(p_values[feat_idx]):
            continue

        p_path = 'data/p_{}.npy'.format(feat_idx)

        # Check if we have already computed this p-value
        if os.path.exists(p_path):
            p_values[feat_idx] = np.load(p_path)
            print('p-value for {}: {}'.format(ccle_features[feat_idx], p_values[feat_idx]))
            continue
        
        # Check if the model assigns zero weight to this feature such that it can be ignored
        if model_weights[feat_idx] == 0:
            # print('p-value for {}: 1 (0 weight in model)'.format(ccle_features[feat_idx]))
            p_values[feat_idx] = 1
            continue
    
        print('************ Missing p-value for {} ************'.format(feat_idx))

    # Save the aggregated results
    np.save(all_p_path, p_values)

    # Print the top-ranked features by their heuristic weight
    ntop = 10
    top_features = np.argsort(np.abs(model_weights))[::-1][:ntop]
    for rank, (feat_idx, importance) in enumerate(zip(top_features, np.abs(model_weights)[top_features])):
        p_value = p_values[feat_idx]
        print('{}. {} importance={:.4f} p={:.4f}'.format(rank+1, ccle_features[feat_idx].replace('_MUT',' Mut'), importance, p_value))

    if np.any(np.isnan(p_values)):
        print('{} NaN p-values!'.format(np.isnan(p_values).sum()))
        missing = np.where(np.isnan(p_values))[0]
        print(missing)
        print([ccle_features[idx] for idx in missing])
        print('Setting to 1')
        p_values[np.isnan(p_values)] = 1

    # Only consider features with substantial heuristic feature importance
    important = np.abs(model_weights) >= importance_threshold
    p_important = p_values[important]

    print('{} features above importance threshold'.format(important.sum()))
    
    # Multiple testing correction via Benjamini-Hochberg at a 20% FDR
    # discoveries = bh(p_values, fdr_threshold) # Test all features
    discoveries = np.arange(len(ccle_features))[important][bh(p_important, fdr_threshold)] # Test important features
    discovery_genes = ccle_features[discoveries]
    discovery_p = p_values[discoveries]
    discovery_weights = model_weights[discoveries]
    
    # Get the heuristic ranking of the discoveries
    discovery_ranks = np.zeros(len(ccle_features))
    discovery_ranks[np.argsort(np.abs(model_weights))[::-1]] = np.arange(len(ccle_features))+1
    discovery_ranks = discovery_ranks[discoveries].astype(int)

    # Print the discoveries along with their model weights and p-values
    order = np.argsort(np.abs(discovery_weights))[::-1]
    print('')
    print('Gene & Model Weight & Rank & $p$-value \\\\')
    for r,g,w,p in zip(discovery_ranks[order],
                     discovery_genes[order],
                     discovery_weights[order],
                     discovery_p[order]):

        print('{} & {} & {:.4f} & {:.4f} \\\\'.format(g.replace('_MUT', ' Mut'),r,w,p))


