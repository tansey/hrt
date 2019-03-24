import os
import numpy as np
import torch
from sim_liang import load_or_create_dataset
from sim_predictors import ModelInfo, get_model, get_conditional
from pyhrt.knockoffs import empirical_risk_knockoffs


def run(trial):
    N = 500 # total number of samples
    P = 500 # number of features
    S = 40 # number of signal features
    T = 100 # test sample size
    fdr_threshold = 0.1

    X, y, truth = load_or_create_dataset(trial, N, P, S)
    np.random.seed(trial*P)

    infos = [ModelInfo(trial, 'Partial Least Squares', fit_pls, 'pls'),
             ModelInfo(trial, 'Lasso', fit_lasso_cv, 'lasso'),
             ModelInfo(trial, 'Elastic Net', fit_elastic_net_cv, 'enet'),
             ModelInfo(trial, 'Bayesian Ridge', fit_bridge, 'bridge'),
             ModelInfo(trial, 'Polynomial Kernel Ridge', fit_kridge, 'kridge'),
             ModelInfo(trial, 'RBF Support Vector', fit_svr, 'svr'),
             ModelInfo(trial, 'Random Forest', fit_forest, 'rf') 
               ]

    folds = get_model(infos[0], X, y, None, False).folds
    models = [get_model(info, X, y, folds, False) for info in infos]

    # Generate a null sample for each feature
    X_null = np.zeros_like(X)
    for j in range(X.shape[1]):
        # Load the conditional model for this feature
        conditional = get_conditional(trial, j)

        # Draw a sample from it
        X_null[:,j], _ = conditional()

    for info, model in zip(infos, models):
        print('\tRunning ERK for {}'.format(info.name))

        # Create the model-specific test statistic (MSE)
        tstat = lambda X_target: ((y - model.predict(X_target))**2).mean()

        # Run the knockoffs procedure
        selected, knockoff_stats = empirical_risk_knockoffs(X,  X_null=X_null)

        np.save('data/{}/{}_selected.npy'.format(trial, info.prefix), selected)
        np.save('data/{}/{}_knockoff_stats.npy'.format(trial, info.prefix), knockoff_stats)


if __name__ == '__main__':
    for trial in range(100):
        print('Trial {}'.format(trial+1))
        run(trial)







