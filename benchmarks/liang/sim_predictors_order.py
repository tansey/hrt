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

def accumulation_test(pvals, hfun, alpha=0.2, numerator_plus=0, denominator_plus=0):
    n = len(pvals)
    pvals = pvals.clip(1e-10,1-1e-10)
    fdp_est = (numerator_plus+np.cumsum(hfun(pvals))) / (denominator_plus+1.+np.arange(n))
    fdp_est_vs_alpha = np.where(fdp_est <= alpha)[0]
    return max(fdp_est_vs_alpha) if len(fdp_est_vs_alpha) > 0 else -1

def hinge_exp(pvals, alpha=0.2, C=2):
    return accumulation_test(pvals, lambda x: C*np.log(1. / (C * (1-x)))*(x > (1 - 1./C)), alpha=alpha)


def seq_step_plus(pvals, alpha=0.1, C=2):
    return accumulation_test(pvals, lambda x: C * (x > (1 - 1. / float(C))),
                alpha=alpha, numerator_plus=C, denominator_plus=1)


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

def order_vs_p_scatter(orders, p, n, c):
    plt.close()
    with sns.axes_style('white', {'legend.frameon': True}):
        plt.rc('font', weight='bold')
        plt.rc('grid', lw=3)
        plt.rc('lines', lw=2)
        plt.rc('axes', lw=2)
        ntests = len(order[0])
        buckets = np.zeros((len(order), ntests))
        for trial_idx, (trial_order, trial_p) in enumerate(zip(order, p)):
            buckets[trial_idx, trial_order] = trial_p
        plt.plot(np.arange(ntests)+1, buckets.mean(axis=0), color=c)
        plt.fill_between(np.arange(ntests)+1,
                         buckets.mean(axis=0) - buckets.std(axis=0),
                         buckets.mean(axis=0) + buckets.std(axis=0),
                         color=c,
                         alpha=0.7)
        plt.xlabel('Heuristic rank', fontsize=18, weight='bold')
        plt.ylabel('$\\mathbf{p}$-value', fontsize=18, weight='bold')
        
def linear_model_heuristic(models):
    '''Order the features by their average coefficient weight in the model'''
    w = [m.coef_ for m in models]
    return np.argsort(np.abs(np.mean(w, axis=0)))[::-1]

def rf_heuristic(models):
    w = [m.feature_importances_ for m in models]
    return np.argsort(np.abs(np.mean(w, axis=0)))[::-1]
    
if __name__ == '__main__':
    N = 500 # total number of samples
    P = 500 # number of features
    S = 40 # number of signal features
    nperms = 5000
    ntrials = 100
    fdr_threshold = 0.1
    nfolds = 5
    
    p_values = defaultdict(lambda: np.full((ntrials, P), np.nan))
    orders = defaultdict(lambda: np.zeros((ntrials, P), dtype=int))
    tpr_vals = defaultdict(lambda: np.full(ntrials, np.nan))
    fdr_vals = defaultdict(lambda: np.full(ntrials, np.nan))
    for trial in range(ntrials):
        print(trial)
        TRUTH_PATH = 'data/{}/truth.csv'.format(trial)
        truth = np.loadtxt(TRUTH_PATH, delimiter=',')

        infos = [ModelInfo(trial, 'Lasso', None, 'lasso'),
                 ModelInfo(trial, 'Elastic Net', None, 'enet'),
                 ModelInfo(trial, 'Bayesian Ridge', None, 'bridge'),
                 ModelInfo(trial, 'Random Forest', None, 'rf')
                   ]

        models = [get_model(info, None, None, None, False) for info in infos]

        # Load the p-values for the predictor models
        for info, model in zip(infos, models):
            # Get the heuristic ordering of the models
            if info.name == 'Random Forest':
                order = rf_heuristic(model.models)
            else:
                order = linear_model_heuristic(model.models)
            orders[info.name][trial] = order

            # Get the p-values and add correction term
            all_p_filename = 'data/{}/{}.npy'.format(trial, info.prefix)
            p_values[info.name][trial] = np.load(all_p_filename)
            p_values[info.name][trial] = (p_values[info.name][trial] * nperms + 1) / (nperms+1)
            
            # Run the ordered testing procedure
            p_ordered = p_values[info.name][trial][order]
            selected = order[:seq_step_plus(p_ordered, alpha=fdr_threshold, C=2)+1]
            pred = np.zeros(P, dtype=bool)
            pred[selected] = True
            tpr_vals[info.name][trial] = tpr(truth, pred)
            fdr_vals[info.name][trial] = fdr(truth, pred)

    # print('Plotting signal p-values')
    # signal_p_values = [signal_p_values[label] for label in labels]
    # p_plot(signal_p_values, labels)
    # plt.savefig('plots/predictors.pdf', bbox_inches='tight')
    # plt.close()
    labels = [info.name for info in infos]
    print('Plotting order vs p-value')
    for info, order, p, n, c in zip(infos,
                              [orders[label] for label in labels],
                              [p_values[label] for label in labels],
                              labels,
                              ['blue', 'orange', 'green', 'purple']):
        order_vs_p_scatter(order, p, n, c)
        plt.savefig('plots/predictors-order-vs-p-{}.pdf'.format(info.prefix), bbox_inches='tight')
        plt.close()

    print('Plotting power and FDR results')
    results_plot([tpr_vals[label] for label in labels],
                 [fdr_vals[label] for label in labels],
                 [label.replace(' ', '\n') for label in labels],
                 fdr_threshold)
    plt.savefig('plots/predictors-ordered.pdf', bbox_inches='tight')
    plt.close()


