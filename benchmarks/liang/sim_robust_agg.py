import matplotlib
matplotlib.use('Agg')
import sys
import os
import numpy as np
import matplotlib.pylab as plt
import seaborn as sns
from pyhrt.utils import bh_predictions, tpr, fdr, pretty_str


def p_plot(p_values, intervals, start=0, end=1):
    plt.close()
    with sns.axes_style('white', {'legend.frameon': True}):
        plt.rc('font', weight='bold')
        plt.rc('grid', lw=3)
        plt.rc('lines', lw=2)
        plt.rc('axes', lw=2)
        styles = ['--', '-.']
        for idx, interval in enumerate(intervals):
            p = p_values[:,:,idx,idx].flatten()
            p = p[~np.isnan(p)]
            p = np.sort(p)
            x = np.concatenate([[0],p,[1]])
            y = np.concatenate([[0],(np.arange(p.shape[0])+1.)/p.shape[0],[1]])
            plt.plot(x, y, label='[{},{}]'.format(50-interval, 50+interval), lw=2, ls=styles[idx//6])
        plt.plot([0,1], [0,1], color='black', ls='--', lw=3, label='U(0,1)', alpha=0.7)
        plt.xlim([start,end])
        plt.ylim([start,end])
        plt.xlabel('p-value', fontsize=18, weight='bold')
        plt.ylabel('Empirical CDF', fontsize=18, weight='bold')
        plt.legend(loc='upper left', ncol=2)

def p_tpr_fdr(p_values, intervals, truth, fdr_threshold=0.1):
    interval_tpr = np.full((intervals.shape[0], intervals.shape[0]), np.nan)
    interval_fdr = np.full(interval_tpr.shape, np.nan)
    for idx1, interval in enumerate(intervals):
        for idx2, interval in enumerate(intervals):
            p = p_values[:,idx1,idx2]
            if np.any(np.isnan(p)):
                continue
            p_indices = ~np.isnan(p)
            p = p[p_indices]
            t = truth[p_indices]
            d = bh_predictions(p, fdr_threshold)
            interval_tpr[idx1,idx2] = tpr(t,d)
            interval_fdr[idx1,idx2] = fdr(t,d)
    return interval_tpr, interval_fdr

def pct_plot(intervals, vals, center=None):
    plt.close()
    with sns.axes_style('white', {'legend.frameon': True}):
        plt.rc('font', weight='bold')
        plt.rc('grid', lw=3)
        plt.rc('lines', lw=2)
        plt.rc('axes', lw=2)
        ax = sns.heatmap(vals, cmap='plasma', xticklabels=50+intervals, yticklabels=50-intervals, center=center)
        ax.invert_yaxis()
        plt.ylabel('Lower interval', fontsize=18, weight='bold')
        plt.xlabel('Upper interval', fontsize=18, weight='bold')


def main():
    N = 500 # total number of samples
    P = 500 # number of features
    S = 40 # number of signal features
    nperms = 5000
    nbootstraps = 100
    intervals = np.array([0,5,10,15,20,25,30,35,40,45])
    fdr_threshold = 0.1
    ntrials = 100
    linear_p_values = np.full((ntrials, P, intervals.shape[0], intervals.shape[0]), np.nan)
    nonlinear_p_values = np.copy(linear_p_values)
    linear_tpr, linear_fdr = np.zeros((ntrials, intervals.shape[0], intervals.shape[0])), np.zeros((ntrials, intervals.shape[0], intervals.shape[0]))
    nonlinear_tpr, nonlinear_fdr = np.zeros((ntrials, intervals.shape[0], intervals.shape[0])), np.zeros((ntrials, intervals.shape[0], intervals.shape[0]))

    for trial in range(ntrials):
        print('trial: {}'.format(trial))
        TRUTH_PATH = 'data/{}/truth.csv'.format(trial)
        truth = np.loadtxt(TRUTH_PATH, delimiter=',')

        LINEAR_P_VALUE_PATH = 'data/{}/sweep_robust_linear_p_values.npy'.format(trial)
        if os.path.exists(LINEAR_P_VALUE_PATH):
            linear_p_values[trial] = np.load(LINEAR_P_VALUE_PATH)
        
        if np.any(np.isnan(linear_p_values[trial])):
            for feature in range(P):
                if np.any(np.isnan(linear_p_values[trial, feature])):
                    P_LINEAR_PATH = 'data/{}/sweep_robust_linear_p_value_{}.npy'.format(trial, feature)
                    linear_p_values[trial,feature] = np.load(P_LINEAR_PATH) if os.path.exists(P_LINEAR_PATH) else np.nan
            if not np.any(np.isnan(linear_p_values[trial])):
                np.save(LINEAR_P_VALUE_PATH, linear_p_values[trial])


        NONLINEAR_P_VALUE_PATH = 'data/{}/sweep_robust_nonlinear_p_values.npy'.format(trial)
        if os.path.exists(NONLINEAR_P_VALUE_PATH):
            nonlinear_p_values[trial] = np.load(NONLINEAR_P_VALUE_PATH)
        
        if np.any(np.isnan(nonlinear_p_values[trial])):
            for feature in range(P):
                if np.any(np.isnan(nonlinear_p_values[trial,feature])):
                    P_NONLINEAR_PATH = 'data/{}/sweep_robust_nonlinear_p_value_{}.npy'.format(trial, feature)
                    nonlinear_p_values[trial,feature] = np.load(P_NONLINEAR_PATH) if os.path.exists(P_NONLINEAR_PATH) else np.nan
            if not np.any(np.isnan(nonlinear_p_values[trial])):
                np.save(NONLINEAR_P_VALUE_PATH, nonlinear_p_values[trial])

        linear_p_values[trial] = linear_p_values[trial] * nperms / (nperms+1)
        nonlinear_p_values[trial] = nonlinear_p_values[trial] * nperms / (nperms+1)
        linear_tpr[trial], linear_fdr[trial] = p_tpr_fdr(linear_p_values[trial], intervals, truth, fdr_threshold)
        nonlinear_tpr[trial], nonlinear_fdr[trial] = p_tpr_fdr(nonlinear_p_values[trial], intervals, truth, fdr_threshold)
        # print(pretty_str(nonlinear_p_values[:,0,0]))
        # print(pretty_str(nonlinear_fdr))

    print('*** Linear model ({} samples) ***'.format(np.sum([np.all(~np.isnan(t)) for t in linear_tpr])))
    print('TPR:\n{} std: {}'.format(pretty_str(np.nanmean(linear_tpr, axis=0)*100), pretty_str(np.nanstd(linear_tpr, axis=0)*100)))
    print('FDR:\n{} std: {}'.format(pretty_str(np.nanmean(linear_fdr, axis=0)*100), pretty_str(np.nanstd(linear_fdr, axis=0)*100)))
    print('')
    print('*** Nonlinear model ({} samples) ***'.format(np.sum([np.all(~np.isnan(t)) for t in nonlinear_tpr])))
    print('TPR:\n{} std: {}'.format(pretty_str(np.nanmean(nonlinear_tpr, axis=0)*100), pretty_str(np.nanstd(nonlinear_tpr, axis=0)*100)))
    print('FDR:\n{} std: {}'.format(pretty_str(np.nanmean(nonlinear_fdr, axis=0)*100), pretty_str(np.nanstd(nonlinear_fdr, axis=0)*100)))
    print('')
    
    pct_plot(intervals, np.nanmean(nonlinear_tpr, axis=0))
    plt.savefig('plots/liang-sweep-tpr.pdf', bbox_inches='tight')
    pct_plot(intervals, np.nanmean(nonlinear_fdr, axis=0), center=fdr_threshold)
    plt.savefig('plots/liang-sweep-fdr.pdf', bbox_inches='tight')

    p_plot(linear_p_values[:,:S], intervals)
    plt.savefig('plots/liang-linear-alternative-p-values.pdf', bbox_inches='tight')
    p_plot(nonlinear_p_values[:,:S], intervals)
    plt.savefig('plots/liang-nonlinear-alternative-p-values.pdf', bbox_inches='tight')
    p_plot(linear_p_values[:,S:], intervals)
    plt.savefig('plots/liang-linear-null-p-values.pdf', bbox_inches='tight')
    p_plot(nonlinear_p_values[:,S:], intervals)
    plt.savefig('plots/liang-nonlinear-null-p-values.pdf', bbox_inches='tight')

    p_plot(linear_p_values[:,:S], intervals, start=0, end=0.1)
    plt.savefig('plots/liang-linear-alternative-p-values-zoomed.pdf', bbox_inches='tight')
    p_plot(nonlinear_p_values[:,:S], intervals, start=0, end=0.1)
    plt.savefig('plots/liang-nonlinear-alternative-p-values-zoomed.pdf', bbox_inches='tight')
    p_plot(linear_p_values[:,S:], intervals, start=0, end=0.1)
    plt.savefig('plots/liang-linear-null-p-values-zoomed.pdf', bbox_inches='tight')
    p_plot(nonlinear_p_values[:,S:], intervals, start=0, end=0.1)
    plt.savefig('plots/liang-nonlinear-null-p-values-zoomed.pdf', bbox_inches='tight')

if __name__ == '__main__':
    main()
