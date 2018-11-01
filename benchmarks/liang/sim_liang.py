import sys
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pylab as plt
import torch
from sim_model import fit_nn, fit_cv, PermutationConditional
from pyhrt.hrt import hrt, holdout_permutation_test
from pyhrt.utils import bh_predictions, tpr, fdr


def liang_sim_xy(N=500, P=500):
    '''Generates data from the simulation study in Liang et al, JASA 2017'''
    X = (np.random.normal(size=(N,1)) + np.random.normal(size=(N,P))) / 2.
    w00 = 1
    w10 = 2
    w11 = 1
    w21 = 2
    w30 = 2
    y = w00*X[:,0] + w10*np.tanh(w11*X[:,1] + w21*X[:,2]) + w30*X[:,3] + np.random.normal(0, 0.5, size=N)
    # Return X, y, and the binary true discovery labels
    return X, y, np.concatenate([np.ones(4), np.zeros(P-4)])

def generalized_liang_sim_xy(N=500, P=500, S=100):
    '''Generates data from a generalized version of the simulation study in
    Liang et al, JASA 2017. For every pair of variables from 1:S, the result
    is a non-linearity (tanh) applied to a weighted sum of the pair.'''
    X = (np.random.normal(size=(N,1)) + np.random.normal(size=(N,P))) / 2.
    w0 = np.random.normal(1, size=S//4)
    w1 = np.random.normal(2, size=S//4)
    w2 = np.random.normal(2, size=S//4)
    w21 = np.random.normal(1, size=(1,S//4))
    w22 = np.random.normal(2, size=(1,S//4))
    y = X[:,0:S:4].dot(w0) + X[:,1:S:4].dot(w1) + np.tanh(w21*X[:,2:S:4] + w22*X[:,3:S:4]).dot(w2) + np.random.normal(0, 0.5, size=N)
    # Return X, y, and the binary true discovery labels
    return X, y, np.concatenate([np.ones(S), np.zeros(P-S)])

def switching_liang_sim_xy(N=500, P=100, R=5):
    '''Generates data from the simulation study in Liang et al, JASA 2017'''
    X = (np.random.normal(size=(N,1)) + np.random.normal(size=(N,P-R))) / 2.
    r = np.random.choice(R, replace=True, size=N)
    w0 = np.random.normal(1, size=R)
    w1 = np.random.normal(2, size=R)
    w2 = np.random.normal(2, size=R)
    w21 = np.random.normal(1, size=R)
    w22 = np.random.normal(2, size=R)
    y = np.zeros(N)
    Z = np.zeros((N,R))
    truth = np.zeros((R,P-R), dtype=int)
    for i in range(R):
        y[r == i] = (X[r == i,i*4] * w0[i] + 
                     X[r == i,i*4+1] * w1[i] +
                     w2[i] * np.tanh(w21[i]*X[r==i,i*4+2] + w22[i]*X[r==i,i*4+3]))
        Z[r==i, i] = 1
        truth[i] = np.concatenate([np.zeros(i*4), np.ones(4), np.zeros((R-i-1)*4), np.zeros(P-5*R)])
    y += np.random.normal(0, 0.5, size=N)
    X = np.concatenate([X,Z], axis=1)
    # Return X, y, and the binary true discovery labels
    return X, y, truth

def load_or_create(path, P):
    if os.path.exists(path + '.npy'):
        return np.load(path + '.npy')
    a = np.full(P, np.nan)
    np.save(path, a)
    return a

def load_or_create_dataset(trial, N, P, S):
    X_PATH = 'data/{}/X.csv'.format(trial)
    Y_PATH = 'data/{}/Y.csv'.format(trial)
    TRUTH_PATH = 'data/{}/truth.csv'.format(trial)
    TRIAL_PATH = 'data/{}'.format(trial)
    if os.path.exists(X_PATH):
        X = np.loadtxt(X_PATH, delimiter=',')
        y = np.loadtxt(Y_PATH, delimiter=',')
        truth = np.loadtxt(TRUTH_PATH, delimiter=',')
    else:
        # Generate the data
        # X, y, truth = liang_sim_xy(N, P)
        X, y, truth = generalized_liang_sim_xy(N, P, S)
        # X, y, truth = switching_liang_sim_xy(N, P, S)

        if not os.path.exists(TRIAL_PATH):
            os.makedirs(TRIAL_PATH)
        np.savetxt(X_PATH, X, delimiter=',')
        np.savetxt(Y_PATH, y, delimiter=',')
        np.savetxt(TRUTH_PATH, truth, delimiter=',')
    return X, y, truth

def run(trial, feature, reset, cv, robust):
    N = 500 # total number of samples
    P = 500 # number of features
    S = 40 # number of signal features
    T = 100 # test sample size
    nperms = 5000
    fdr_threshold = 0.1
    
    model_prefix = 'cv_' if cv else ''
    p_prefix = 'cv_' if cv else ''
    p_prefix += 'robust_' if robust else ''
    nbootstraps = 100 if robust else 1
    LINEAR_PATH = 'data/{}/{}linear.pt'.format(trial, model_prefix)
    NONLINEAR_PATH = 'data/{}/{}nonlinear.pt'.format(trial, model_prefix)
    P_PERM_PATH = 'data/{}/{}perm_p_values'.format(trial, p_prefix)
    P_LINEAR_PATH = 'data/{}/{}linear_p_values'.format(trial, p_prefix)
    P_NONLINEAR_PATH = 'data/{}/{}nonlinear_p_values'.format(trial, p_prefix)
    Pi_PERM_PATH = 'data/{}/{}perm_p_values_{}'.format(trial, p_prefix, feature)
    Pi_LINEAR_PATH = 'data/{}/{}linear_p_values_{}'.format(trial, p_prefix, feature)
    Pi_NONLINEAR_PATH = 'data/{}/{}nonlinear_p_values_{}'.format(trial, p_prefix, feature)
    BOUNDS_LINEAR_PATH = 'data/{}/{}linear_bounds_{}'.format(trial, p_prefix, feature)
    BOUNDS_NONLINEAR_PATH = 'data/{}/{}nonlinear_bounds_{}'.format(trial, p_prefix, feature)
    CONDITIONAL_PATH = 'data/{}/conditional_{}{}.pt'.format(trial, p_prefix, feature)

    X, y, truth = load_or_create_dataset(trial, N, P, S)

    # Load the checkpoint if available
    if not reset and os.path.exists(LINEAR_PATH):
        linear_model = torch.load(LINEAR_PATH)
        nonlinear_model = torch.load(NONLINEAR_PATH)
    else:
        # Train the model
        print('Fitting models with N={} P={} S={} T={} nperms={}'.format(N, P, S, T, nperms))
        sys.stdout.flush()
        if cv:
            print('Using CV models')
            linear_model = fit_cv(X, y, verbose=False, model_type='linear')
            nonlinear_model = fit_cv(X, y, verbose=False, model_type='nonlinear')
        else:
            print('Using holdout models')
            linear_model = fit_nn(X[:-T], y[:-T], verbose=False, model_type='linear')
            nonlinear_model = fit_nn(X[:-T], y[:-T], verbose=False, model_type='nonlinear')
        torch.save(linear_model, LINEAR_PATH)
        torch.save(nonlinear_model, NONLINEAR_PATH)

    # Track all the p-values
    perm_p_values = load_or_create(P_PERM_PATH, P) if not robust else None
    linear_p_values = load_or_create(P_LINEAR_PATH, P)
    nonlinear_p_values = load_or_create(P_NONLINEAR_PATH, P)

    # test statistics for the two models
    y_train = y if cv else y[:-T]
    y_test = y if cv else y[-T:]
    X_train = X if cv else X[:-T]
    X_test = None if cv else X[-T:]
    tstat_linear = lambda X_target: ((y_test - linear_model.predict(X_target))**2).mean()
    tstat_nonlinear = lambda X_target: ((y_test - nonlinear_model.predict(X_target))**2).mean()

    if trial == 0:
        import matplotlib
        matplotlib.use('Agg')
        import seaborn as sns
        with sns.axes_style('white', {'legend.frameon': True}):
            plt.rc('font', weight='bold')
            plt.rc('grid', lw=3)
            plt.rc('lines', lw=2)
            plt.rc('axes', lw=2)
            plt.scatter(y_train,nonlinear_model.predict(X_train))
            plt.plot([y.min(), y.max()], [y.min(), y.max()], color='red', ls='--')
            plt.xlabel('Truth', fontsize=18, weight='bold')
            plt.ylabel('Predicted', fontsize=18, weight='bold')
            plt.savefig('plots/liang-nonlinear-fit{}.pdf'.format('-cv' if cv else ''), bbox_inches='tight')
            plt.close()

            plt.rc('font', weight='bold')
            plt.rc('grid', lw=3)
            plt.rc('lines', lw=2)
            plt.rc('axes', lw=2)
            plt.scatter(y_train,linear_model.predict(X_train))
            plt.plot([y.min(), y.max()], [y.min(), y.max()], color='red', ls='--')
            plt.xlabel('Truth', fontsize=18, weight='bold')
            plt.ylabel('Predicted', fontsize=18, weight='bold')
            plt.savefig('plots/liang-linear-fit{}.pdf'.format('-cv' if cv else ''), bbox_inches='tight')
            plt.close()
    
    conditional = None
    lower = None
    upper = None
    perm_folds = nonlinear_model.folds if cv else None
    print('Feature: {}'.format(feature))
    
    if not robust:
        print('Running permutation test')
        if np.isnan(perm_p_values[feature]) and not os.path.exists(Pi_PERM_PATH + '.npy'):
            permer = PermutationConditional(X if cv else X[-T:], feature, perm_folds)
            perm_p_value = hrt(feature, tstat_nonlinear, X_train, X_test=X_test, nperms=nperms, conditional=permer)['p_value']
            np.save(Pi_PERM_PATH, perm_p_value)
            print('Trial {} feature {} {} {} permutation p={}'.format(trial, feature, 'robust' if robust else '', 'cv' if cv else '', perm_p_value))
    
    print('Running linear HRT')
    if np.isnan(linear_p_values[feature]) and not os.path.exists(Pi_LINEAR_PATH + '.npy'):
        linear_results = hrt(feature, tstat_linear, X_train, X_test=X_test, nperms=nperms, nbootstraps=nbootstraps, conditional=conditional)
        linear_p_value = linear_results['p_value']
        conditional = linear_results['sampler']
        np.save(Pi_LINEAR_PATH, linear_p_value)
        print('Trial {} feature {} {} {} linear hrt p={}'.format(trial, feature, 'robust' if robust else '', 'cv' if cv else '', linear_p_value))
        if robust:
            lower = linear_results['lower']
            upper = linear_results['upper']
            np.save(BOUNDS_LINEAR_PATH, np.concatenate([lower, upper]))

    print('Running nonlinear HRT')
    if np.isnan(nonlinear_p_values[feature]) and not os.path.exists(Pi_NONLINEAR_PATH + '.npy'):
        nonlinear_results = hrt(feature, tstat_nonlinear, X_train, X_test=X_test, nperms=nperms,
                                     nbootstraps=nbootstraps, conditional=conditional,
                                     lower=lower, upper=upper)
        nonlinear_p_value = nonlinear_results['p_value']
        np.save(Pi_NONLINEAR_PATH, nonlinear_p_value)
        torch.save(nonlinear_results['sampler'], CONDITIONAL_PATH)
        print('Trial {} feature {} {} {} nonlinear hrt p={}'.format(trial, feature, 'robust' if robust else '', 'cv' if cv else '', nonlinear_p_value))
        if robust:
            lower = nonlinear_results['lower']
            upper = nonlinear_results['upper']
            np.save(BOUNDS_NONLINEAR_PATH, np.concatenate([lower, upper]))
        

    print('')
    print('Done!')
    sys.stdout.flush()

if __name__ == '__main__':
    trial = int(sys.argv[1])
    feature = int(sys.argv[2])
    reset = len(sys.argv) > 3 and '--reset' in sys.argv[3:]
    cv = len(sys.argv) > 3 and '--cv' in sys.argv[3:]
    robust = len(sys.argv) > 3 and '--robust' in sys.argv[3:]
    run(trial, feature, reset, cv, robust)


