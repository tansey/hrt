import sys
import os
import numpy as np
import matplotlib.pylab as plt
import torch
import sklearn
import shap


def main():
    N = 500 # total number of samples
    P = 500 # number of features
    S = 40 # number of signal features
    T = 100 # test sample size
    nperms = 5000
    fdr_threshold = 0.1
    trial = int(sys.argv[1])
    sample = int(sys.argv[2])
    TRIAL_PATH = 'data/{}'.format(trial)
    X_PATH = 'data/{}/X.csv'.format(trial)
    Y_PATH = 'data/{}/Y.csv'.format(trial)
    TRUTH_PATH = 'data/{}/truth.csv'.format(trial)
    NONLINEAR_PATH = 'data/{}/cv_nonlinear.pt'.format(trial)
    SHAP_PATH = 'data/{}/shap_values'.format(trial)
    SHAPi_PATH = 'data/{}/shap_values_{}'.format(trial, sample)

    # Load the data and CV model
    X = np.loadtxt(X_PATH, delimiter=',')
    y = np.loadtxt(Y_PATH, delimiter=',')
    truth = np.loadtxt(TRUTH_PATH, delimiter=',')
    nonlinear_model = torch.load(NONLINEAR_PATH)
    yhat = nonlinear_model.predict(X)

    # Check if all of the results have already been generated and compiled
    if os.path.exists(SHAP_PATH + '.npy'):
        all_shap_values = np.load(SHAP_PATH + '.npy')
        if not np.any(np.isnan(all_shap_values[sample])):
            return

    # Check if this sample has already been generated
    if os.path.exists(SHAPi_PATH + '.npy'):
        shap_values = np.load(SHAPi_PATH + '.npy')
        if not np.any(np.isnan(shap_values)):
            return

    # Run the shapley model for each fold
    for fold, model in zip(nonlinear_model.folds, nonlinear_model.models):
        if sample not in fold:
            continue
        train_mask = np.ones(N, dtype=bool)
        train_mask[fold] = False
        X_train = X[train_mask]
        # use Kernel SHAP to explain test set predictions
        explainer = shap.KernelExplainer(model.predict, X_train)
        shap_values = explainer.shap_values(X[0:1], nsamples=500)
        np.save(SHAPi_PATH, shap_values)
        print(shap_values)
        break



if __name__ == '__main__':
    main()


