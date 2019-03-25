import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
import torch
from collections import defaultdict
from sim_model import fit_cv
from sim_predictors import ModelInfo, get_r2, CvModel, PLSPredictor
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

def liang_r2(trial):
    nn_r2_path = 'data/{}/nn_r2'.format(trial)
    ols_r2_path = 'data/{}/ols_r2'.format(trial)
    if os.path.exists(nn_r2_path + '.npy'):
        return np.load(ols_r2_path + '.npy'), np.load(nn_r2_path + '.npy')
    from sklearn.metrics import r2_score
    from sim_liang import load_or_create_dataset
    X, y, truth = load_or_create_dataset(trial, None, None, None)
    LINEAR_PATH = 'data/{}/cv_linear.pt'.format(trial)
    NONLINEAR_PATH = 'data/{}/cv_nonlinear.pt'.format(trial)
    linear_model = torch.load(LINEAR_PATH)
    nonlinear_model = torch.load(NONLINEAR_PATH)
    y_ols, y_nn = linear_model.predict(X), nonlinear_model.predict(X)
    ols_score, nn_score = r2_score(y, y_ols), r2_score(y, y_nn)
    np.save(ols_r2_path, ols_score)
    np.save(nn_r2_path, nn_score)
    return ols_score, nn_score


def results_plot(tpr_vals, fdr_vals, r2_vals, names, fdr_threshold):
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
        order = np.argsort([np.mean(r) if len(r) > 0 else 1 for r in r2_vals])
        # for t, f, r, n in zip(tpr_vals, fdr_vals, r2_vals, names):
        for idx in order:
            t, f, r, n = tpr_vals[idx], fdr_vals[idx], r2_vals[idx], names[idx]
            # rates.extend(r)
            rates.extend(t)
            rates.extend(f)
            # labels.extend(['$\\mathbf{r^2}$']*len(r))
            labels.extend(['TPR']*len(t))
            labels.extend(['FDR']*len(f))
            models.extend([n]*(len(t)+len(f)))#+len(r)
        df = pd.DataFrame({'value': rates, 'Rate': labels, 'Model': models})
        df['value'] = df['value'].astype(float)
        ax = sns.boxplot(x='Model', y='value', hue='Rate', data=df)  # RUN PLOT
        plt.xlabel('', fontsize=18, weight='bold')
        # plt.ylabel('Power, FDR, and $\\mathbf{r^2}$', fontsize=18, weight='bold')
        plt.ylabel('Power and FDR', fontsize=18, weight='bold')
        plt.axhline(fdr_threshold, color='red', lw=2, ls='--')
        # ax.tick_params(labelsize=10)
        plt.legend(loc='upper right')
        sns.despine(offset=10, trim=True)

def r2_scatter(tpr_vals, r2_vals, names):
    plt.close()
    with sns.axes_style('white', {'legend.frameon': True}):
        plt.rc('font', weight='bold')
        plt.rc('grid', lw=3)
        plt.rc('lines', lw=2)
        plt.rc('axes', lw=2)
        # order = np.argsort([np.mean(r) if len(r) > 0 else 1 for r in r2_vals])
        # for idx in order:
            # plt.scatter(r2_vals[idx], tpr_vals[idx], label=names[idx])
        for t, r, n in zip(tpr_vals, r2_vals, names):
            plt.scatter(r, t, label=n)
        plt.xlabel('Cross-validation $\\mathbf{r^2}$', fontsize=18, weight='bold')
        plt.ylabel('Power', fontsize=18, weight='bold')
        plt.xlim([0.8, 1])
        plt.xticks([0.8, 0.85, 0.9, 0.95, 1], ['0.8', '0.85', '0.9', '0.95', '1'])
        # plt.legend(loc='upper left', ncol=2)
        

    
if __name__ == '__main__':
    N = 500 # total number of samples
    P = 500 # number of features
    S = 40 # number of signal features
    nperms = 5000
    ntrials = 100
    fdr_threshold = 0.1
    nfolds = 5
    
    p_values = defaultdict(lambda: np.full((ntrials, P), np.nan))
    tpr_vals = defaultdict(lambda: np.full(ntrials, np.nan))
    fdr_vals = defaultdict(lambda: np.full(ntrials, np.nan))
    signal_p_values = defaultdict(list)
    r2_scores = defaultdict(list)
    knockoff_tpr = defaultdict(lambda: np.full(ntrials, np.nan))
    knockoff_fdr = defaultdict(lambda: np.full(ntrials, np.nan))
    for trial in range(ntrials):
        print(trial)
        TRUTH_PATH = 'data/{}/truth.csv'.format(trial)
        truth = np.loadtxt(TRUTH_PATH, delimiter=',')

        infos = [ModelInfo(trial, 'PLS', None, 'pls'),
                 ModelInfo(trial, 'Lasso', None, 'lasso'),
                 ModelInfo(trial, 'Elastic Net', None, 'enet'),
                 ModelInfo(trial, 'Bayesian Ridge', None, 'bridge'),
                 ModelInfo(trial, 'Kernel Ridge', None, 'kridge'),
                 ModelInfo(trial, 'RBF Support Vector', None, 'svr'),
                 ModelInfo(trial, 'Random Forest', None, 'rf') 
                   ]

        # Load the p-values for the predictor models
        for info in infos:
            r2_scores[info.name].append(get_r2(trial, info))
            all_p_filename = 'data/{}/{}.npy'.format(trial, info.prefix)
            if not os.path.exists(all_p_filename):
                np.save(all_p_filename, np.full(P, np.nan))
            p_values[info.name][trial] = np.load(all_p_filename)
            for feature in range(P):
                p_filename = 'data/{}/{}_{}.npy'.format(trial, info.prefix, feature)
                if np.isnan(p_values[info.name][trial,feature]) and os.path.exists(p_filename):
                    p = np.load(p_filename)
                    p_values[info.name][trial,feature] = p
            np.save(all_p_filename, p_values[info.name][trial])
            p_values[info.name][trial] = (p_values[info.name][trial] * nperms + 1) / (nperms+1)
            missing = np.isnan(p_values[info.name][trial])
            signal_p_values[info.name].extend(p_values[info.name][trial,:S][~missing[:S]])
            p_trial = np.ones(P)
            p_trial[~missing] = p_values[info.name][trial][~missing]
            pred = bh_predictions(p_trial, fdr_threshold)
            tpr_vals[info.name][trial] = tpr(truth, pred)
            fdr_vals[info.name][trial] = fdr(truth, pred)

            if np.any(missing):
                if missing.sum() > 10:
                    print('Total missing: {}'.format(missing.sum()))
                else:
                    print('Missing: {}'.format(np.arange(P)[missing]))

        # Load the p-values for the other models
        for path, name in [('cv_robust_nonlinear_p_values', 'Neural Net'), ('cv_robust_linear_p_values', 'OLS')]:
            p_trial = np.load('data/{}/{}.npy'.format(trial, path))
            p_trial = (p_trial * nperms + 1) / (nperms+1)
            signal_p_values[name].extend(p_trial[:S])

            missing = np.isnan(p_trial)
            p_trial[missing] = 1
            pred = bh_predictions(p_trial[~missing], fdr_threshold)
            tpr_vals[name][trial] = tpr(truth, pred)
            fdr_vals[name][trial] = fdr(truth, pred)

        # Load the r^2 values for the other models
        ols_r2, nn_r2 = liang_r2(trial)
        r2_scores['Neural Net'].append(nn_r2)
        r2_scores['OLS'].append(ols_r2)

        # Load the discoveries for lasso coefficient magnitude knockoffs
        knockoffs = np.loadtxt('data/{}/knockoffs.csv'.format(trial)).astype(int)
        pred = np.zeros(P, dtype=bool)
        pred[knockoffs-1] = True
        tpr_vals['Lasso Knockoffs'][trial] = tpr(truth, pred)
        fdr_vals['Lasso Knockoffs'][trial] = fdr(truth, pred)

        # Load the empirical risk knockoff selections for all the models
        infos.append(ModelInfo(trial, 'OLS', None, 'linear'))
        infos.append(ModelInfo(trial, 'Neural Net', None, 'nonlinear'))
        for info in infos:
            selected_file = 'data/{}/{}_selected.npy'.format(trial, info.prefix)
            if not os.path.exists(selected_file):
                print('Trial {} missing {} knockoffs. Skipping...'.format(trial, info.name))
                continue
            pred = np.zeros(P, dtype=bool)
            pred[np.load(selected_file).astype(int)] = True
            knockoff_tpr[info.name][trial] = tpr(truth, pred)
            knockoff_fdr[info.name][trial] = fdr(truth, pred)

    print('Plotting signal p-values')
    labels = ['OLS'] + [info.name for info in infos] + ['Neural Net']
    signal_p_values = [signal_p_values[label] for label in labels]
    p_plot(signal_p_values, labels)
    plt.savefig('plots/predictors.pdf', bbox_inches='tight')
    plt.close()

    print('Plotting r^2 vs power')
    r2_scatter([tpr_vals[label] for label in labels],
                 [r2_scores[label] for label in labels],
                 labels)
    plt.savefig('plots/predictors-r2-tpr.pdf', bbox_inches='tight')
    plt.close()


    print('Plotting HRT power and FDR results vs lasso knockoffs')
    labels = labels + ['Lasso Knockoffs']
    results_plot([tpr_vals[label] for label in labels],
                 [fdr_vals[label] for label in labels],
                 [r2_scores[label] for label in labels],
                 [label.replace(' ', '\n') for label in labels],
                 fdr_threshold)
    plt.savefig('plots/predictors-tpr-fdr.pdf', bbox_inches='tight')
    plt.close()

    print('Plotting ERK power and FDR results vs lasso knockoffs')
    labels = labels + ['Lasso Knockoffs']
    results_plot([knockoff_tpr[label] for label in labels],
                 [knockoff_fdr[label] for label in labels],
                 [r2_scores[label] for label in labels],
                 [label.replace(' ', '\n') for label in labels],
                 fdr_threshold)
    plt.savefig('plots/knockoffs-tpr-fdr.pdf', bbox_inches='tight')
    plt.close()


