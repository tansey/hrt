from __future__ import print_function
import numpy as np

def empirical_risk_knockoffs(X, tstat_fn, fdr,
                             conditionals=None,
                             X_null=None,
                             verbose=False, save_nulls=False, control_mfdr=True):
    '''Runs an empirical risk knockoff procedure that performs simultaneous
    feature selection with FDR control, only needing to evaluate the model
    once per feature. If control_mfdr=False, the model controls the true FDR;
    by default, we only control the so-called modified FDR (mFDR).'''
    knockoff_stats = np.zeros(X.shape[1])
    t_true = tstat_fn(X)

    if X_null is None:
        if conditionals is None:
            raise Exception('Must provide either null samples or complete conditionals')
        # Generate the nulls from the conditionals
        X_null = np.zeros_like(X)
        for j in range(X.shape[1]):
            X_null[:,j] = conditionals[j]()

    for j in range(X.shape[1]):
        X_temp = np.copy(X)
        X_temp[:,j] = X_null[:,j]

        # Get the test-statistic under the null
        knockoff_stats[j] = tstat_fn(X_temp)

        if verbose:
            print('\t\tfeature {}) t_true: {} t_null: {}'.format(j+1, t_true, knockoff_stats[j]))

    knockoff_stats -= t_true

    # Perform the knockoffs selection procedure
    offset = 0 if control_mfdr else 1
    order = np.argsort(np.abs(knockoff_stats))
    selected = np.array([])
    for t in np.abs(knockoff_stats[order]):
        ratio = ((knockoff_stats <= -t).sum() + offset) / max(1,(knockoff_stats >= t).sum())
        if verbose:
            print('# <= {}: {}'.format(-t, (knockoff_stats <= -t).sum()))
            print('# >= {}: {}'.format(t, (knockoff_stats >= t).sum()))
            print('Ratio: {}'.format(ratio))
            print()
        if ratio <= fdr:
            selected = np.arange(X.shape[1])[knockoff_stats >= t]
            break
    if verbose:
        print('Selected: {}'.format(selected))

    # Return the null samples if desired
    if save_nulls:
        return selected, knockoff_stats, X_null
    return selected, knockoff_stats




