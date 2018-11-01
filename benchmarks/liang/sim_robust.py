# import matplotlib
# matplotlib.use('Agg')
import sys
import os
import numpy as np
# import matplotlib.pylab as plt
# import seaborn as sns
import torch
from sim_model import fit_cv
from pyhrt.hrt import hrt
from pyhrt.utils import bh_predictions, tpr, fdr, pretty_str

def load_or_create(path, P, intervals):
    if os.path.exists(path + '.npy'):
        return np.load(path + '.npy')
    a = np.full((P, intervals.shape[0], intervals.shape[0]), np.nan)
    np.save(path, a)
    return a


def main():
    N = 500 # total number of samples
    P = 500 # number of features
    S = 40 # number of signal features
    nperms = 5000
    nbootstraps = 100
    fdr_threshold = 0.1
    trial = int(sys.argv[1])
    feature = int(sys.argv[2])
    intervals = np.array([0,5,10,15,20,25,30,35,40,45])
    lower, upper = (50 - intervals), 50 + intervals
    reset_models = len(sys.argv) > 3 and '--reset-models' in sys.argv[3:]
    TRIAL_PATH = 'data/{}'.format(trial)
    X_PATH = 'data/{}/X.csv'.format(trial)
    Y_PATH = 'data/{}/Y.csv'.format(trial)
    TRUTH_PATH = 'data/{}/truth.csv'.format(trial)
    LINEAR_PATH = 'data/{}/cv_linear.pt'.format(trial)
    NONLINEAR_PATH = 'data/{}/cv_nonlinear.pt'.format(trial)
    P_LINEAR_PATH = 'data/{}/sweep_robust_linear_p_values'.format(trial)
    P_NONLINEAR_PATH = 'data/{}/sweep_robust_nonlinear_p_values'.format(trial)
    Pi_LINEAR_PATH = 'data/{}/sweep_robust_linear_p_value_{}'.format(trial, feature)
    Pi_NONLINEAR_PATH = 'data/{}/sweep_robust_nonlinear_p_value_{}'.format(trial, feature)

    X = np.loadtxt(X_PATH, delimiter=',')
    y = np.loadtxt(Y_PATH, delimiter=',')
    truth = np.loadtxt(TRUTH_PATH, delimiter=',')

    if reset_models:
        print('Fitting models with N={} P={} S={} nperms={}'.format(N, P, S, nperms))
        sys.stdout.flush()
        linear_model = fit_cv(X, y, verbose=False, model_type='linear')
        nonlinear_model = fit_cv(X, y, verbose=False, model_type='nonlinear')
        torch.save(linear_model, LINEAR_PATH)
        torch.save(nonlinear_model, NONLINEAR_PATH)
    else:
        linear_model = torch.load(LINEAR_PATH)
        nonlinear_model = torch.load(NONLINEAR_PATH)

    linear_p_values = load_or_create(P_LINEAR_PATH, P, intervals)
    nonlinear_p_values = load_or_create(P_NONLINEAR_PATH, P, intervals)
        
    print('Testing with N={} P={} S={} nperms={} nbootstraps={} interval=[{},{}]'.format(N, P, S, nperms, nbootstraps, lower, upper))
    
    # test statistics for the two models
    tstat_linear = lambda X_test: ((y - linear_model.predict(X_test))**2).mean()
    tstat_nonlinear = lambda X_test: ((y - nonlinear_model.predict(X_test))**2).mean()

    print('Feature: {}'.format(feature))

    conditional = None
    linear_p_value = linear_p_values[feature]
    if np.any(np.isnan(linear_p_value)) and os.path.exists(Pi_LINEAR_PATH + '.npy'):
        linear_p_value = np.load(Pi_LINEAR_PATH + '.npy')
    if np.any(np.isnan(linear_p_value)):
        print('Running linear robust CVR test')
        linear_results = hrt(feature, tstat_linear, X, nperms=nperms, nbootstraps=nbootstraps, conditional=conditional, lower=lower, upper=upper)
        # Get the results and reuse the conditional model
        linear_p_value = linear_results['p_value']
        conditional = linear_results['sampler']
        np.save(Pi_LINEAR_PATH, linear_p_value)

    
    nonlinear_p_value = nonlinear_p_values[feature]
    if np.any(np.isnan(nonlinear_p_value)) and os.path.exists(Pi_NONLINEAR_PATH + '.npy'):
        nonlinear_p_value = np.load(Pi_NONLINEAR_PATH + '.npy')
    if np.any(np.isnan(nonlinear_p_value)):
        print('Running nonlinear robust CVR test')
        nonlinear_results = hrt(feature, tstat_nonlinear, X, nperms=nperms, nbootstraps=nbootstraps, conditional=conditional, lower=lower, upper=upper)
        nonlinear_p_value = nonlinear_results['p_value']
        np.save(Pi_NONLINEAR_PATH, nonlinear_p_value)

    print('p-values Robust CVR (linear): {}\nRobust CVR (nonlinear): {}'.format(pretty_str(linear_p_value), pretty_str(nonlinear_p_value)))
    # print('t-weights Robust CVR (linear): {}\nRobust CVR (nonlinear): {}'.format(pretty_str(linear_results['t_weights'] / linear_results['t_weights'].mean()), pretty_str(nonlinear_results['t_weights']/nonlinear_results['t_weights'].mean())))

    # linear_predictions = bh_predictions(linear_p_values, fdr_threshold)
    # nonlinear_predictions = bh_predictions(nonlinear_p_values, fdr_threshold)

    # linear_tpr = tpr(truth, linear_predictions)
    # linear_fdr = fdr(truth, linear_predictions)
    # nonlinear_tpr = tpr(truth, nonlinear_predictions)
    # nonlinear_fdr = fdr(truth, nonlinear_predictions)


    # print('Robust cross-validation randomization test (linear)')
    # print('TPR: {:.2f}%'.format(linear_tpr*100))
    # print('FDR: {:.2f}%'.format(linear_fdr*100))
    # print('')
    # sys.stdout.flush()

    # print('Robust cross-validation randomization test (nonlinear)')
    # print('TPR: {:.2f}%'.format(nonlinear_tpr*100))
    # print('FDR: {:.2f}%'.format(nonlinear_fdr*100))
    # print('')
    # sys.stdout.flush()

    # if trial == 0:
    #     with sns.axes_style('white', {'legend.frameon': True}):
    #         plt.rc('font', weight='bold')
    #         plt.rc('grid', lw=3)
    #         plt.rc('lines', lw=2)
    #         plt.rc('axes', lw=2)
    #         plt.scatter(np.arange(P), linear_p_values, color='red', label='Linear CVR test')
    #         plt.scatter(np.arange(P), nonlinear_p_values, color='blue', label='Non-linear CVR test')
    #         plt.axvline(S + 0.5, ls='--', color='black')
    #         plt.xlabel('Feature index', fontsize=18, weight='bold')
    #         plt.ylabel('p-value', fontsize=18, weight='bold')
    #         legend_props = {'weight': 'bold', 'size': 14}
    #         plt.legend(loc='upper right', prop=legend_props)
    #         plt.savefig('plots/liang-p-values-cv.pdf', bbox_inches='tight')
    #         plt.close()

    #         plt.scatter(linear_p_values[:S], nonlinear_p_values[:S], color='orange', label='True signals')
    #         plt.scatter(linear_p_values[S:], nonlinear_p_values[S:], color='gray', label='True nulls')
    #         plt.xlabel('Linear CVR p-values', fontsize=18, weight='bold')
    #         plt.ylabel('Non-linear CVR p-values', fontsize=18, weight='bold')
    #         plt.plot([0,1],[0,1], color='blue')
    #         legend_props = {'weight': 'bold', 'size': 14}
    #         plt.legend(loc='upper left', prop=legend_props)
    #         plt.savefig('plots/liang-linear-vs-nonlinear-p-values-cv.pdf', bbox_inches='tight')
    #         plt.close()

    print('Done!')
    sys.stdout.flush()

if __name__ == '__main__':
    main()


