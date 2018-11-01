import os
import numpy as np
import torch
from sim_liang import load_or_create_dataset
from sklearn.externals import joblib
from pyhrt.utils import create_folds
from pyhrt.hrt import hrt

class PLSPredictor:
    def __init__(self, m):
        self.m = m
    def predict(self, X):
        return self.m.predict(X)[:,0]

def fit_pls(X, y, ncomponents=10):
    from sklearn.cross_decomposition import PLSRegression
    pls = PLSRegression(n_components=ncomponents)
    pls.fit(X, y)
    return PLSPredictor(pls)

# Run lasso regression for each drug, choosing lambda via 10-fold CV
def fit_lasso_cv(X, y, nfolds=5):
    from sklearn.linear_model import LassoCV
    lasso = LassoCV(cv=nfolds)
    lasso.fit(X,y)
    return lasso

# Run elastic net regression for each drug, choosing lambda via 10-fold CV
def fit_elastic_net_cv(X, y, nfolds=5):
    from sklearn.linear_model import ElasticNetCV
    # The parameter l1_ratio corresponds to alpha in the glmnet R package 
    # while alpha corresponds to the lambda parameter in glmnet
    enet = ElasticNetCV(l1_ratio=np.linspace(0.01, 1.0, 20),
                        alphas=np.exp(np.linspace(-6, 5, 20)),
                        cv=nfolds)
    enet.fit(X,y)
    return enet

def fit_forest(X, y, nestimators=100):
    from sklearn.ensemble import RandomForestRegressor
    rf = RandomForestRegressor(n_estimators=nestimators, n_jobs=8)
    rf.fit(X,y)
    return rf

def fit_extratrees(X, y, nestimators=100):
    from sklearn.ensemble import ExtraTreesRegressor
    rf = ExtraTreesRegressor(n_estimators=nestimators, n_jobs=8)
    rf.fit(X,y)
    return rf

def fit_kridge(X, y):
    from sklearn.kernel_ridge import KernelRidge
    kr = KernelRidge(kernel='poly')
    kr.fit(X,y)
    return kr

def fit_svr(X, y):
    from sklearn.svm import SVR
    svr = SVR(kernel='rbf')
    svr.fit(X, y)
    return svr

def fit_bridge(X, y):
    from sklearn.linear_model import BayesianRidge
    br = BayesianRidge()
    br.fit(X,y)
    return br

def fit_cv(X, y, folds, fit_fn, selected=None):
    models = []
    for fold_idx, fold in enumerate(folds):
        mask = np.ones(X.shape[0], dtype=bool)
        mask[fold] = False
        print('\tFold {} ({} samples)'.format(fold_idx, X[mask].shape[0]))
        X_fold = X[mask]
        if selected is not None:
            X_fold = X_fold[:,selected]
        models.append(fit_fn(X_fold, y[mask]))
    return models

'''Simple model to do CV HRT testing'''
class CvModel:
    def __init__(self, models, folds, name, selected=None):
        self.models = models
        self.folds = folds
        self.name = name
        self.selected = selected

    def predict(self, X):
        y = np.zeros(X.shape[0])
        for fold, model in zip(self.folds, self.models):
            X_fold = X[fold]
            if self.selected is not None:
                X_fold = X_fold[:,self.selected]
            y[fold] = model.predict(X_fold)
        return y

class ModelInfo:
    def __init__(self, trial, name, fit_fn, prefix):
        self.trial = trial
        self.name = name
        self.fit_fn = fit_fn
        self.prefix = prefix
        self.path = 'data/{}/{}_model.joblib'.format(trial, prefix)

def get_model(info, X, y, folds, reset):
    # import torch
    print('Getting {} model'.format(info.name))
    if not reset and os.path.exists(info.path):
        return joblib.load(info.path)
        # return torch.load(info.path)
    model = CvModel(fit_cv(X, y, folds, info.fit_fn), folds, info.name)
    joblib.dump(model, info.path)
    # torch.save(model, info.path)
    return model

def get_conditional(trial, feature):
    import torch
    CONDITIONAL_PATH = 'data/{}/conditional_cv_robust_{}.pt'.format(trial, feature)
    return torch.load(CONDITIONAL_PATH)

def get_r2(trial, info):
    r2_path = 'data/{}/{}_r2'.format(trial, info.prefix)
    if os.path.exists(r2_path + '.npy'):
        return np.load(r2_path + '.npy')
    from sklearn.metrics import r2_score
    X, y, truth = load_or_create_dataset(trial, None, None, None)
    model = get_model(info, X, y, None, False)
    y_pred = model.predict(X)
    score = r2_score(y, y_pred)
    np.save(r2_path, score)
    return score

def run(trial, feature, reset=False):
    N = 500 # total number of samples
    P = 500 # number of features
    S = 40 # number of signal features
    nperms = 5000
    fdr_threshold = 0.1
    nfolds = 5

    X, y, truth = load_or_create_dataset(trial, N, P, S)

    np.random.seed(trial*P+feature)
    
    infos = [ModelInfo(trial, 'Partial Least Squares', fit_pls, 'pls'),
             ModelInfo(trial, 'Lasso', fit_lasso_cv, 'lasso'),
             ModelInfo(trial, 'Elastic Net', fit_elastic_net_cv, 'enet'),
             ModelInfo(trial, 'Bayesian Ridge', fit_bridge, 'bridge'),
             ModelInfo(trial, 'Polynomial Kernel Ridge', fit_kridge, 'kridge'),
             ModelInfo(trial, 'RBF Support Vector', fit_svr, 'svr'),
             ModelInfo(trial, 'Random Forest', fit_forest, 'rf') 
             # ModelInfo(trial, 'Extra Trees', fit_extratrees, 'xtrees')
               ]

    folds = get_model(infos[0], X, y, create_folds(X, nfolds), reset).folds
    models = [get_model(info, X, y, folds, reset) for info in infos]

    # Create the test statistic for each model
    # tstats = [(lambda X_target: ((y - model.predict(X_target))**2).mean()) for model in models]

    # Load the conditional model for this feature
    conditional = get_conditional(trial, feature)

    # Run the normal CVRT for the first model, but save the null samples to
    # avoid recomputing them for the rest of the models.
    info, model = infos[0], models[0]
    tstat = lambda X_target: ((y - model.predict(X_target))**2).mean()
    print('Running CVRT for {}'.format(info.name))
    results = hrt(feature, tstat, X, nperms=nperms,
                            conditional=conditional,
                            lower=conditional.quantiles[0],
                            upper=conditional.quantiles[1],
                            save_nulls=True)
    p_value = results['p_value']
    print('p={}'.format(p_value))
    np.save('data/{}/{}_{}.npy'.format(trial, info.prefix, feature), p_value)

    # Get the relevant values from the full CVRT on the first model
    t_true = results['t_stat']
    X_nulls = results['samples_null']
    quantile_nulls = results['quantiles_null']

    # Run the CVRTs for the remaining models using the same null samples
    X_null = np.copy(X)
    for info, model in zip(infos[1:], models[1:]):
        print('Running cached CVRT for {}'.format(info.name))
        t_weights = np.full(nperms, np.nan)
        t_null = np.full(nperms, np.nan)
        tstat = lambda X_target: ((y - model.predict(X_target))**2).mean()
        t_true = tstat(X)
        for perm in range(nperms):
            if (perm % 500) == 0:
                print('Trial {}'.format(perm))

            # Get the test-statistic under the null
            X_null[:,feature] = X_nulls[perm]
            t_null[perm] = tstat(X_null)
            if t_null[perm] <= t_true:
                # Over-estimate the likelihood
                t_weights[perm] = quantile_nulls[perm,1]
            else:
                # Under-estimate the likelihood
                t_weights[perm] = quantile_nulls[perm,0]

        p_value = t_weights[t_null <= t_true].sum() / t_weights.sum()
        print('p={}'.format(p_value))
        np.save('data/{}/{}_{}.npy'.format(trial, info.prefix, feature), p_value)

def run_parallel(x):
    trial = x[0]
    feature = x[1]
    reset = x[2]
    torch.set_num_threads(1) # bad torch, no biscuit
    try:
        run(trial, feature, reset=reset)
    except:
        pass

def seed_fn():
    np.random.seed()


if __name__ == '__main__':
    import sys
    if len(sys.argv) == 3:
        run(int(sys.argv[1]), int(sys.argv[2]), reset=False)
    else:
        import argparse
        from multiprocessing import Pool
        torch.set_num_threads(1) # bad torch, no biscuit

        parser = argparse.ArgumentParser(description='Worker script for the benchmark.')

        # Experiment settings
        parser.add_argument('min_trial', type=int, help='Min Trial ID')
        parser.add_argument('max_trial', type=int, help='Max Trial ID')
        parser.add_argument('min_feature', type=int, help='Min Feature ID')
        parser.add_argument('max_feature', type=int, help='Max Feature ID')
        parser.add_argument('--reset', action='store_true', help='If specified, resets the models.')
        parser.add_argument('--nthreads', type=int, default=8, help='Number of parallel workers to run.')
        args = parser.parse_args()

        # Build all the jobs
        jobs = []
        for f in range(args.min_feature, args.max_feature+1):
            for t in range(args.min_trial, args.max_trial+1):
                jobs.append((t,f,args.reset))

        # Run in parallel (okay this is processes not threads, but who's counting?)
        with Pool(args.nthreads, initializer=seed_fn) as p:
            p.map(run_parallel, jobs)




