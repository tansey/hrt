import sys
import os
import numpy as np
from pyhrt.utils import bh_predictions, tpr, fdr, pretty_str

def add_if_finished(trial, all_p_values, all_tpr, all_fdr, truth, fdr_threshold):
    if not np.any(np.isnan(all_p_values[trial])):
        predictions = bh_predictions(all_p_values[trial], fdr_threshold)
        all_tpr[trial] = tpr(truth, predictions)
        all_fdr[trial] = fdr(truth, predictions)

if __name__ == '__main__':
    ntrials = 100
    N = 500 # total number of samples
    P = 500 # number of features
    S = 40 # number of signal features
    T = 100 # test sample size
    fdr_threshold = 0.1
    intervals = np.array([0,5,10,15,20,25,30,35,40,45])

    # Simple permutation-style test results
    perm_p_values, linear_p_values, nonlinear_p_values = np.full((ntrials, P),np.nan), np.full((ntrials, P), np.nan), np.full((ntrials, P), np.nan)
    cv_perm_p_values, cv_linear_p_values, cv_nonlinear_p_values = np.full((ntrials, P),np.nan), np.full((ntrials, P),np.nan), np.full((ntrials, P), np.nan)
    (perm_tpr, perm_fdr,
        linear_tpr, linear_fdr,
        nonlinear_tpr, nonlinear_fdr,
        cv_perm_tpr, cv_perm_fdr,
        cv_linear_tpr, cv_linear_fdr,
        cv_nonlinear_tpr, cv_nonlinear_fdr) = (np.full(ntrials, np.nan) for _ in range(12))

    # Robust testing results
    sweep_robust_linear_p_values = np.full((ntrials, P, intervals.shape[0], intervals.shape[0]), np.nan)
    sweep_robust_nonlinear_p_values = np.copy(sweep_robust_linear_p_values)
    (sweep_robust_linear_tpr, sweep_robust_linear_fdr,
    sweep_robust_nonlinear_tpr, sweep_robust_nonlinear_fdr) = (np.full((ntrials, intervals.shape[0], intervals.shape[0]), np.nan) for _ in range(4))
    for trial in range(ntrials):
        if (trial % 25) == 0:
            print(trial)
        TRUTH_PATH = 'data/{}/truth.csv'.format(trial)
        P_PERM_PATH = 'data/{}/perm_p_values.npy'.format(trial)
        P_LINEAR_PATH = 'data/{}/linear_p_values.npy'.format(trial)
        P_NONLINEAR_PATH = 'data/{}/nonlinear_p_values.npy'.format(trial)
        CV_P_PERM_PATH = 'data/{}/cv_perm_p_values.npy'.format(trial)
        CV_P_LINEAR_PATH = 'data/{}/cv_linear_p_values.npy'.format(trial)
        CV_P_NONLINEAR_PATH = 'data/{}/cv_nonlinear_p_values.npy'.format(trial)
        SWEEP_ROBUST_CV_P_LINEAR_PATH = 'data/{}/sweep_robust_linear_p_values.npy'.format(trial)
        SWEEP_ROBUST_CV_P_NONLINEAR_PATH = 'data/{}/sweep_robust_nonlinear_p_values.npy'.format(trial)

        try:
            truth = np.loadtxt(TRUTH_PATH, delimiter=',')
            if os.path.exists(P_PERM_PATH):
                perm_p_values[trial] = np.load(P_PERM_PATH)
            if os.path.exists(P_LINEAR_PATH):
                linear_p_values[trial] = np.load(P_LINEAR_PATH)
            if os.path.exists(P_NONLINEAR_PATH):
                nonlinear_p_values[trial] = np.load(P_NONLINEAR_PATH)
            if os.path.exists(CV_P_PERM_PATH):
                cv_perm_p_values[trial] = np.load(CV_P_PERM_PATH)
            if os.path.exists(CV_P_LINEAR_PATH):
                cv_linear_p_values[trial] = np.load(CV_P_LINEAR_PATH)
            if os.path.exists(CV_P_NONLINEAR_PATH):
                cv_nonlinear_p_values[trial] = np.load(CV_P_NONLINEAR_PATH)
            if os.path.exists(SWEEP_ROBUST_CV_P_LINEAR_PATH):
                sweep_robust_linear_p_values[trial] = np.load(SWEEP_ROBUST_CV_P_LINEAR_PATH)
            if os.path.exists(SWEEP_ROBUST_CV_P_NONLINEAR_PATH):
                sweep_robust_nonlinear_p_values[trial] = np.load(SWEEP_ROBUST_CV_P_NONLINEAR_PATH)
        except Exception as ex:
            print('Trial {}, skipping: {}'.format(trial, ex))
            continue

        add_if_finished(trial, perm_p_values, perm_tpr, perm_fdr, truth, fdr_threshold)
        add_if_finished(trial, linear_p_values, linear_tpr, linear_fdr, truth, fdr_threshold)
        add_if_finished(trial, nonlinear_p_values, nonlinear_tpr, nonlinear_fdr, truth, fdr_threshold)
        add_if_finished(trial, cv_perm_p_values, cv_perm_tpr, cv_perm_fdr, truth, fdr_threshold)
        add_if_finished(trial, cv_linear_p_values, cv_linear_tpr, cv_linear_fdr, truth, fdr_threshold)
        add_if_finished(trial, cv_nonlinear_p_values, cv_nonlinear_tpr, cv_nonlinear_fdr, truth, fdr_threshold)
        for i in range(intervals.shape[0]):
            for j in range(intervals.shape[0]):
                add_if_finished(trial, sweep_robust_linear_p_values[:,:,i,j], sweep_robust_linear_tpr[:,i,j], sweep_robust_linear_fdr[:,i,j], truth, fdr_threshold)
                add_if_finished(trial, sweep_robust_nonlinear_p_values[:,:,i,j], sweep_robust_nonlinear_tpr[:,i,j], sweep_robust_nonlinear_fdr[:,i,j], truth, fdr_threshold)

    print('Permutation test ({} trials)'.format((~np.isnan(perm_tpr)).sum()))
    print('TPR: {:.2f}%'.format(np.nanmean(perm_tpr)*100))
    print('FDR: {:.2f}%'.format(np.nanmean(perm_fdr)*100))
    print('')
    sys.stdout.flush()

    print('Heldout data randomization test (linear) ({} trials)'.format((~np.isnan(linear_tpr)).sum()))
    print('TPR: {:.2f}%'.format(np.nanmean(linear_tpr)*100))
    print('FDR: {:.2f}%'.format(np.nanmean(linear_fdr)*100))
    print('')
    sys.stdout.flush()

    print('Heldout data randomization test (nonlinear) ({} trials)'.format((~np.isnan(nonlinear_tpr)).sum()))
    print('TPR: {:.2f}%'.format(np.nanmean(nonlinear_tpr)*100))
    print('FDR: {:.2f}%'.format(np.nanmean(nonlinear_fdr)*100))
    print('')
    sys.stdout.flush()

    print('Cross-validation permutation test ({} trials)'.format((~np.isnan(cv_perm_tpr)).sum()))
    print('TPR: {:.2f}%'.format(np.nanmean(cv_perm_tpr)*100))
    print('FDR: {:.2f}%'.format(np.nanmean(cv_perm_fdr)*100))
    print('')
    sys.stdout.flush()

    print('Cross-validation randomization test (linear) ({} trials)'.format((~np.isnan(cv_linear_tpr)).sum()))
    print('TPR: {:.2f}%'.format(np.nanmean(cv_linear_tpr)*100))
    print('FDR: {:.2f}%'.format(np.nanmean(cv_linear_fdr)*100))
    print('')
    sys.stdout.flush()

    print('Cross-validation randomization test (nonlinear) ({} trials)'.format((~np.isnan(cv_nonlinear_tpr)).sum()))
    print('TPR: {:.2f}%'.format(np.nanmean(cv_nonlinear_tpr)*100))
    print('FDR: {:.2f}%'.format(np.nanmean(cv_nonlinear_fdr)*100))
    print('')
    sys.stdout.flush()

    print('Robust Cross-validation randomization test (linear) ({} trials)'.format((~np.isnan(sweep_robust_linear_tpr)).sum(axis=0)[0,0]))
    print('TPR:\n{}'.format(pretty_str(np.nanmean(sweep_robust_linear_tpr, axis=0)*100)))
    print('FDR:\n{}'.format(pretty_str(np.nanmean(sweep_robust_linear_fdr, axis=0)*100)))
    print('')
    sys.stdout.flush()

    print('Cross-validation randomization test (nonlinear) ({} trials)'.format((~np.isnan(sweep_robust_nonlinear_tpr)).sum(axis=0)[0,0]))
    print('TPR:\n{}'.format(pretty_str(np.nanmean(sweep_robust_nonlinear_tpr, axis=0)*100)))
    print('FDR:\n{}'.format(pretty_str(np.nanmean(sweep_robust_nonlinear_fdr, axis=0)*100)))
    print('')
    sys.stdout.flush()

