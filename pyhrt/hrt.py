from __future__ import print_function
import sys
import numpy as np
from pyhrt.continuous import calibrate_continuous
from pyhrt.discrete import calibrate_discrete

def holdout_permutation_test(feature, tstat_fn, X_train, X_test, nperms=None, verbose=False):
    '''Perform a naive permutation test on heldout data. This is not a valid testing procedure.'''
    N = X_train.shape[0]
    P = X_train.shape[1]

    # Order of magnitude more permutations than data points or features by default
    if nperms is None:
        nperms = max(N*10,P*10)

    # Get the test-statistic using the real data
    if verbose:
        print('Getting true test statistic')
        sys.stdout.flush()
    t_true = tstat_fn(X_test)

    X_null = np.copy(X_test)
    t_null = np.zeros(nperms)
    for perm in range(nperms):
        if verbose and (perm % 50) == 0:
            print('Trial {}'.format(perm))
            sys.stdout.flush()

        # Permute the column
        holdout_indices = np.arange(X_null.shape[0])
        np.random.shuffle(holdout_indices)
        X_null[:,feature] = X_null[holdout_indices,feature]

        # Get the test-statistic under the null
        t_null[perm] = tstat_fn(X_null)

        if verbose:
            print('t_null: {}'.format(t_null[perm]))
            sys.stdout.flush()

    # Calculate the p-value by MC approximation (ignore the need for +1 in the denominator)
    return (t_null <= t_true).mean()

def hrt(feature, tstat_fn, X, X_test=None,
             nperms=None, verbose=False, conditional=None,
             nquantiles=101, nbootstraps=100, nfolds=5,
             ks_threshold=0.005, tv_threshold=0.005, p_threshold=0,
             lower=None, upper=None, save_nulls=False):
    '''Perform the heldout data randomization test. If conditional is not specified, it is inferred from
    the number of unique values, u, in the training set: u <= 0.25*N uses discrete; otherwise, continuous
    is used. If the conditional is not simply discrete or continuous (e.g. ordinal, discrete-continuous
    mixtures, bounded continuous, etc.) the user can specify a custom conditional function.

    For custom functions, no calibration will be performed to make sure the distribution matches-- this is left
    to the custom function. The custom function should return a tuple: (sample, probs) for the test dataset
    feature column, where sample is a random sample and probs is a 3-tuple of [median, lower, upper]
    corresponding to the middle probability estimate and its lower and upper confidence intervals.
    If the sampling model does not produce confidence intervals, one can simply return a 3-tuple of
    all 1s; this will correspond to an HRT under the assumption that the conditional distribution
    is the true distribution.
    '''
    
    # Find the test type automatically
    if conditional is None:
        u = len(np.unique(X[:,feature]))
        conditional = 'discrete' if u <= 0.25*X.shape[0] else 'continuous'
    
    if conditional == 'continuous':
        if verbose:
            print('Running continuous HRT')
        # Fit a robust conditional model for X_j
        X_train = np.concatenate([X,X_test], axis=0) if X_test is not None else X
        results = calibrate_continuous(X_train, feature,
                                   X_test=X_test,
                                   nquantiles=nquantiles,
                                   nbootstraps=nbootstraps,
                                   nfolds=nfolds,
                                   ks_threshold=ks_threshold,
                                   p_threshold=p_threshold)
        conditional = results['sampler']

        # If no quantiles have been specified, use the auto-calibrated ones
        if lower is None:
            lower = np.array([results['lower']])
        if upper is None:
            upper = np.array([results['upper']])
    elif conditional == 'discrete':
        if verbose:
            print('Running discrete HRT')
        # Fit a robust conditional model for X_j
        X_train = np.concatenate([X,X_test], axis=0) if X_test is not None else X
        results = calibrate_discrete(X_train, feature,
                                     X_test=X_test,
                                     nquantiles=nquantiles,
                                     nbootstraps=nbootstraps,
                                     nfolds=nfolds,
                                     tv_threshold=tv_threshold,
                                     p_threshold=p_threshold)
        conditional = results['sampler']

        # If no quantiles have been specified, use the auto-calibrated ones
        if lower is None:
            lower = np.array([results['lower']])
        if upper is None:
            upper = np.array([results['upper']])
    else:
        results = {'sampler': conditional}
        if verbose:
            print('Running HRT with custom conditional model')

    N = X.shape[0]
    P = X.shape[1]

    # Order of magnitude more permutations than data points by default
    if nperms is None:
        nperms = max(N*10,P*10)

    # If no quantiles have been chosen, assume we can use the median
    if lower is None:
        lower = np.array([50])
    if upper is None:
        upper = np.array([50])

    # If we were given scalar quantiles, convert them to 1d arrays
    if np.isscalar(lower):
        lower = np.array([lower])
    if np.isscalar(upper):
        upper = np.array([upper])

    results['lower'] = lower
    results['upper'] = upper

    # Set the quantiles from the bootstrap models
    quantiles = np.concatenate([lower, upper])
    conditional.quantiles = quantiles

    # Get the test-statistic using the real data
    X_null = np.copy(X) if X_test is None else np.copy(X_test)
    t_true = tstat_fn(X_null)
    t_null = np.zeros(nperms)
    quants_null = np.zeros((nperms, quantiles.shape[0]))
    t_weights = np.zeros((nperms, len(lower), len(upper)))
    if save_nulls:
        X_null_samples = np.full((nperms, X.shape[0]), np.nan)
    for perm in range(nperms):
        if (perm % 500) == 0:
            print('Trial {}'.format(perm))
        # Sample from the conditional null model
        X_null[:,feature], quants_null[perm] = conditional()

        # Save the null if desired
        if save_nulls:
            X_null_samples[perm] = X_null[:,feature]

        # Get the test-statistic under the null
        t_null[perm] = tstat_fn(X_null)

        if t_null[perm] <= t_true:
            # Over-estimate the likelihood
            t_weights[perm] = quants_null[perm,len(lower):][np.newaxis,:]
        else:
            # Under-estimate the likelihood
            t_weights[perm] = quants_null[perm,:len(lower)][:,np.newaxis]

        if verbose > 1:
            from utils import pretty_str
            print('t_true: {} t_null: {} weight:\n{}'.format(t_true, t_null[perm], pretty_str(np.exp(t_weights[perm]))))

    # Calculate the weights using a numerically stable approach that accounts for having very small probabilities
    # t_weights = np.exp(t_weights)

    # Calculate the p-value conservatively using the calibrated confidence weights
    results['p_value'] = np.squeeze(t_weights[t_null <= t_true].sum(axis=0) / t_weights.sum(axis=0))
    results['t_stat'] = t_true
    results['t_null'] = t_null
    results['t_weights'] = np.squeeze(t_weights.sum(axis=0))
    results['quantiles_null'] = quants_null
    if save_nulls:
        results['samples_null'] = X_null_samples

    return results


def test_hrt_continuous():
    # Generate the ground truth
    N = 1000
    X = (np.random.normal(size=(N,4)) + np.random.normal(size=(N,1)))/2.
    logits = np.array([np.exp(X[:,0]**2), np.exp(X[:,0]), np.exp(2*X[:,0])]).T
    pi = logits / logits.sum(axis=1, keepdims=True)
    mu = np.array([X[:,0], 5*X[:,1], -2*X[:,1]*X[:,0]]).T
    sigma = np.ones((X.shape[0],3))
    true_gmm = GaussianMixtureModel(pi, mu, sigma)

    # Sample some observations
    y = true_gmm.sample()
    truth = true_gmm.cdf(y)

    # Fit the model
    from continuous import fit_mdn
    print('Fitting predictor')
    split = int(np.round(X.shape[0]*0.8))
    model = fit_mdn(X[:split], y[:split], nepochs=20)
    
    # Use the negative log-likelihood as the test statistic
    tstat = lambda X_test: -np.log(model.predict(X_test).pdf(y[split:])).mean()

    p_values = np.zeros(4)
    for j in range(X.shape[1]):
        p_values[j] = hrt(j, tstat, X[:split], X[split:], nperms=1000)
        print('Feature {}: p={}'.format(j, p_values[j]))

def test_hrt_discrete():
    from discrete import fit_classifier, MultinomialModel
    from utils import ilogit
    # Generate the ground truth
    N = 500
    X = (np.random.random(size=(N,4)) <= ilogit((np.random.normal(size=(N,4)) + np.random.normal(size=(N,1)))/2.)).astype(int)
    true_logits = (np.array([0.5,1,1.5])[np.newaxis,:]*X[:,0:1]
                    + np.array([-2,1,-0.5])[np.newaxis,:]*X[:,1:2]
                    + X[:,0:1] * X[:,1:2] * np.array([-2,1,2])[np.newaxis,:])
    truth = np.exp(true_logits) / np.exp(true_logits).sum(axis=1, keepdims=True)
    true_model = MultinomialModel(truth)
            
    # Sample some observations
    y = true_model.sample()
        
    # Fit the model
    print('Fitting predictor')
    split = int(np.round(X.shape[0]*0.8))
    model = fit_classifier(X[:split], y[:split], nepochs=20)
        
    # Use the negative log-likelihood as the test statistic
    tstat = lambda X_test: -np.log(model.predict(X_test).pmf(y[split:])).mean()

    p_values = np.zeros(4)
    for j in range(X.shape[1]):
        p_values[j] = hrt(j, tstat, X[:split], X[split:], nperms=1000, nbootstraps=10)['p_value']
        print('Feature {}: p={}'.format(j, p_values[j]))

if __name__ == '__main__':
    test_hrt_discrete()
