import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
import torch
from collections import defaultdict
from sim_model import fit_cv
from sim_predictors import ModelInfo, CvModel, PLSPredictor, get_model
from pyhrt.utils import bh_predictions, tpr, fdr, pretty_str


def p_plot(p_values, labels, start=0, end=1):
    plt.close()
    with sns.axes_style('white', {'legend.frameon': True}):
        plt.rc('font', weight='bold')
        plt.rc('grid', lw=3)
        plt.rc('lines', lw=2)
        plt.rc('axes', lw=2)
        plt.rc('xtick', labelsize=14)
        plt.rc('ytick', labelsize=14)
        matplotlib.rcParams['pdf.fonttype'] = 42
        matplotlib.rcParams['ps.fonttype'] = 42
        for p, label in zip(p_values, labels):
            p = np.array(p)
            p = p[~np.isnan(p)]
            p = np.sort(p)
            print(p.shape)
            x = np.concatenate([[0],p,[1]])
            y = np.concatenate([[0],(np.arange(p.shape[0])+1.)/p.shape[0],[1]])
            plt.plot(x, y, label=label, lw=2)
        plt.plot([0,1], [0,1], color='black', ls='--', lw=3, label='U(0,1)', alpha=0.7)
        plt.xlim([start,end])
        plt.ylim([start,end])
        plt.xlabel('p-value', fontsize=18, weight='bold')
        plt.ylabel('Empirical CDF', fontsize=18, weight='bold')
        plt.legend(loc='lower right')

def results_plot(tpr_vals, fdr_vals, names, fdr_threshold):
    import pandas as pd
    plt.close()
    with sns.axes_style('white', {'legend.frameon': True}):
        plt.rc('font', weight='bold')
        plt.rc('grid', lw=3)
        plt.rc('lines', lw=2)
        plt.rc('axes', lw=2)
        plt.figure(figsize=(12,5))
        rates = []
        labels = []
        models = []
        # for t, f, r, n in zip(tpr_vals, fdr_vals, r2_vals, names):
        for t, f, n in zip(tpr_vals, fdr_vals, names):
            rates.extend(t)
            rates.extend(f)
            labels.extend(['TPR']*len(t))
            labels.extend(['FDR']*len(f))
            models.extend([n]*(len(t)+len(f)))#+len(r)
        df = pd.DataFrame({'value': rates, 'Rate': labels, 'Model': models})
        df['value'] = df['value'].astype(float)
        ax = sns.boxplot(x='Model', y='value', hue='Rate', data=df)  # RUN PLOT
        plt.ylabel('Power and FDR', fontsize=18, weight='bold')
        plt.axhline(fdr_threshold, color='red', lw=2, ls='--')
        plt.xlabel('')
        # ax.tick_params(labelsize=10)
        plt.legend(loc='upper right')
        sns.despine(offset=10, trim=True)

def linear_model_importance(models):
    '''Average coefficient weight in the model'''
    return np.abs(np.array([m.coef_ for m in models])).mean(axis=0)

def rf_importance(models):
    '''Average feature importance weight'''
    return np.array([m.feature_importances_ for m in models]).mean(axis=0)
    
if __name__ == '__main__':
    N = 500 # total number of samples
    P = 500 # number of features
    S = 40 # number of signal features
    nperms = 5000
    ntrials = 100
    fdr_threshold = 0.1
    nfolds = 5
    importance_threshold = 1e-3 # Minimum importance for a feature to be considered
    
    p_values = defaultdict(lambda: np.full((ntrials, P), np.nan))
    tpr_vals = defaultdict(lambda: np.full(ntrials, np.nan))
    fdr_vals = defaultdict(lambda: np.full(ntrials, np.nan))
    for trial in range(ntrials):
        print(trial)
        TRUTH_PATH = 'data/{}/truth.csv'.format(trial)
        truth = np.loadtxt(TRUTH_PATH, delimiter=',')

        infos = [ModelInfo(trial, 'Random Forest', None, 'rf'),
                 ModelInfo(trial, 'Bayesian Ridge', None, 'bridge'),
                 ModelInfo(trial, 'Elastic Net', None, 'enet'),
                 ModelInfo(trial, 'Lasso', None, 'lasso')]

        models = [get_model(info, None, None, None, False) for info in infos]

        # Load the p-values for the predictor models
        for info, model in zip(infos, models):
            # Get the heuristic ordering of the models
            if info.name == 'Random Forest':
                importance = rf_importance(model.models)
            else:
                importance = linear_model_importance(model.models)
            
            # Get the p-values and add correction term
            all_p_filename = 'data/{}/{}.npy'.format(trial, info.prefix)
            p_values[info.name][trial] = np.load(all_p_filename)
            p_values[info.name][trial] = (p_values[info.name][trial] * nperms + 1) / (nperms+1)
            
            # Run the filtered testing procedure
            important = importance >= importance_threshold
            p_important = p_values[info.name][trial][important]
            print('\tFiltering down to {} features'.format(important.sum()))
            pred = np.zeros(P, dtype=int)
            pred[important] = bh_predictions(p_important, fdr_threshold)
            tpr_vals[info.name][trial] = tpr(truth, pred)
            fdr_vals[info.name][trial] = fdr(truth, pred)

    labels = [info.name for info in infos]
    
    print('Plotting power and FDR results')
    results_plot([tpr_vals[label] for label in labels],
                 [fdr_vals[label] for label in labels],
                 [label.replace(' ', '\n') for label in labels],
                 fdr_threshold)
    plt.savefig('plots/predictors-importance.pdf', bbox_inches='tight')
    plt.close()


