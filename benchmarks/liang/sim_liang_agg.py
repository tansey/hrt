import matplotlib
matplotlib.use('Agg')
import os
import numpy as np
import matplotlib.pylab as plt
import seaborn as sns
from pyhrt.utils import bh_predictions, tpr, fdr, pretty_str

def p_plot(p_values, S, start=0, end=1):
    plt.close()
    with sns.axes_style('white', {'legend.frameon': True}):
        plt.rc('font', weight='bold')
        plt.rc('grid', lw=3)
        plt.rc('lines', lw=2)
        plt.rc('axes', lw=2)
        for p, label in zip([p_values[:,:S], p_values[:,S:]], ['Alternative', 'Null']):
            p = p.flatten()
            p = p[~np.isnan(p)]
            p = np.sort(p)
            x = np.concatenate([[0],p,[1]])
            y = np.concatenate([[0],(np.arange(p.shape[0])+1.)/p.shape[0],[1]])
            plt.plot(x, y, label=label, lw=2)
        plt.plot([0,1], [0,1], color='black', ls='--', lw=3, label='U(0,1)', alpha=0.7)
        plt.xlim([start,end])
        plt.ylim([start,end])
        plt.xlabel('p-value', fontsize=18, weight='bold')
        plt.ylabel('Empirical CDF', fontsize=18, weight='bold')
        plt.legend(loc='lower right')

def bounds_plot(bounds):
    plt.close()
    with sns.axes_style('white', {'legend.frameon': True}):
        plt.rc('font', weight='bold')
        plt.rc('grid', lw=3)
        plt.rc('lines', lw=2)
        plt.rc('axes', lw=2)
        lower = bounds[:,:,0][~np.isnan(bounds[:,:,0])].flatten()
        upper = bounds[:,:,1][~np.isnan(bounds[:,:,1])].flatten()
        plt.hist(lower, label='Lower band', color='blue', bins=np.linspace(0,50,51), normed=True)
        plt.hist(upper, label='Upper band', color='orange', bins=np.linspace(50,100,51), normed=True)
        plt.xlabel('Band value', fontsize=18, weight='bold')
        plt.ylabel('Proportion', fontsize=18, weight='bold')
        plt.legend(loc='upper right')

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
        for t, f, n in zip(tpr_vals, fdr_vals, names):
            rates.extend(t)
            rates.extend(f)
            labels.extend(['TPR']*len(t))
            labels.extend(['FDR']*len(f))
            models.extend([n]*(len(t)+len(f)))
        df = pd.DataFrame({'value': rates, 'Rate': labels, 'Model': models})
        ax = sns.boxplot(x='Model', y='value', hue='Rate', data=df)  # RUN PLOT
        plt.xlabel('', fontsize=18, weight='bold')
        plt.ylabel('Power and FDR', fontsize=18, weight='bold')
        plt.axhline(fdr_threshold, color='red', lw=2, ls='--')
        # ax.tick_params(labelsize=10)
        plt.legend(loc='upper right')
        sns.despine(offset=10, trim=True)


def main():
    N = 500 # total number of samples
    P = 500 # number of features
    S = 40 # number of signal features
    nperms = 5000
    nbootstraps = 100
    fdr_threshold = 0.1
    ntrials = 100
    names = ['Holdout\nPermutation', 'Calibrated\nHRT\n(linear)', 'Uncalibrated\nHRT',
             'CV\nPermutation', 'Calibrated\nCV-HRT\n(linear)', 'Uncalibrated\nCV-HRT',
             'Calibrated\nHRT\n(linear)', 'Calibrated\nHRT',
             'Calibrated\nCV-HRT\n(linear)', 'Calibrated\nCV-HRT']
    prefixes = ['perm', 'linear', 'nonlinear',
                'cv_perm', 'cv_linear', 'cv_nonlinear',
                'robust_linear', 'robust_nonlinear',
                'cv_robust_linear', 'cv_robust_nonlinear']
    p_values = np.full((len(prefixes), ntrials, P), np.nan)
    tpr_vals, fdr_vals = np.full((len(prefixes), ntrials), np.nan), np.full((len(prefixes), ntrials), np.nan)

    for idx, prefix in enumerate(prefixes):
        for trial in range(ntrials):
            if (trial % 25) == 0:
                print('{} trial: {}'.format(prefix, trial))
            TRUTH_PATH = 'data/{}/truth.csv'.format(trial)
            truth = np.loadtxt(TRUTH_PATH, delimiter=',')

            P_VALUE_PATH = 'data/{}/{}_p_values.npy'.format(trial, prefix)
            if os.path.exists(P_VALUE_PATH):
                p_values[idx, trial] = np.load(P_VALUE_PATH)
        
            clean_up_needed = False
            if np.any(np.isnan(p_values[idx,trial])):
                clean_up_needed = True
                for feature in range(P):
                    Pi_PATH = 'data/{}/{}_p_values_{}.npy'.format(trial, prefix, feature)
                    if np.isnan(p_values[idx, trial, feature]) and os.path.exists(Pi_PATH):
                        try:
                            p_values[idx,trial,feature] = np.load(Pi_PATH)
                        except:
                            os.remove(Pi_PATH)
            
            # p_values[idx, trial] = p_values[idx, trial] * nperms / (nperms+1)
            missing = np.isnan(p_values[idx, trial])
            pred = bh_predictions(p_values[idx, trial][~missing], fdr_threshold)
            tpr_vals[idx, trial] = tpr(truth[~missing], pred)
            fdr_vals[idx, trial] = fdr(truth[~missing], pred)

            if not np.any(np.isnan(p_values[idx,trial])):
                # clean up
                if clean_up_needed:
                    np.save(P_VALUE_PATH, p_values[idx,trial])
                    for feature in range(P):
                        Pi_PATH = 'data/{}/{}_p_values_{}.npy'.format(trial, prefix, feature)
                        if os.path.exists(Pi_PATH):
                            # print('Would delete {}'.format((idx, trial, feature)))
                            os.remove(Pi_PATH)
            else:
                print('Trial {} Nulls: {}'.format(trial, np.where(np.isnan(p_values[idx, trial]))[0]))

        if 'robust' in prefix:
            # Get the distribution of confidence intervals
            bounds = np.full((ntrials, P, 2), np.nan)
            for trial in range(ntrials):
                BOUNDS_PATH = 'data/{}/{}_bounds.npy'.format(trial, prefix)
                if os.path.exists(BOUNDS_PATH):
                    bounds[trial] = np.load(BOUNDS_PATH)
                
                clean_up_needed = False
                if np.any(np.isnan(bounds[trial])):
                    clean_up_needed = True
                    for feature in range(P):
                        BOUNDS_i_PATH = 'data/{}/{}_bounds_{}.npy'.format(trial, prefix, feature)
                        if np.any(np.isnan(bounds[trial, feature])) and os.path.exists(BOUNDS_i_PATH):
                            bounds[trial,feature] = np.load(BOUNDS_i_PATH)
                
                if not np.any(np.isnan(bounds[trial])):
                    # clean up
                    if clean_up_needed:
                        np.save(BOUNDS_PATH, bounds[trial])
                        for feature in range(P):
                            BOUNDS_i_PATH = 'data/{}/{}_bounds_{}.npy'.format(trial, prefix, feature)
                            if os.path.exists(BOUNDS_i_PATH):
                                # print('Would delete {}'.format(BOUNDS_i_PATH))
                                os.remove(BOUNDS_i_PATH)

            bounds_plot(bounds)
            plt.savefig('plots/liang-bounds-{}.pdf'.format(prefix.replace('_', '-')), bbox_inches='tight')


        print('*** {} model ({} trials) ***'.format(names[idx], (~np.isnan(tpr_vals[idx])).sum()))
        print('TPR: {:.2f}%'.format(np.nanmean(tpr_vals[idx], axis=0)*100))
        print('FDR: {:.2f}%'.format(np.nanmean(fdr_vals[idx], axis=0)*100))
        print('')
    
        p_plot(p_values[idx], S)
        plt.savefig('plots/liang-p-values-{}.pdf'.format(prefix.replace('_','-')), bbox_inches='tight')

    selected = np.array([0,3,2,5,7,9])
    results_plot([tpr_vals[i] for i in selected],
                 [fdr_vals[i] for i in selected],
                 [names[i] for i in selected],
                 fdr_threshold)
    plt.savefig('plots/liang-tpr-fdr.pdf', bbox_inches='tight')

if __name__ == '__main__':
    main()



