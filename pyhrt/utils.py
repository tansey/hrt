import numpy as np
from scipy import stats as st

def p2z(p):
    return st.norm.ppf(p)

def fdr(truth, pred, axis=None):
    return ((pred==1) & (truth==0)).sum(axis=axis) / pred.sum(axis=axis).astype(float).clip(1,np.inf)

def tpr(truth, pred, axis=None):
    return ((pred==1) & (truth==1)).sum(axis=axis) / truth.sum(axis=axis).astype(float).clip(1,np.inf)

def true_positives(truth, pred, axis=None):
    return ((pred==1) & (truth==1)).sum(axis=axis)

def false_positives(truth, pred, axis=None):
    return ((pred==1) & (truth==0)).sum(axis=axis)

def ilogit(x):
    return 1. / (1. + np.exp(-x))

def pretty_str(p, decimal_places=2, ignore=None, label_columns=False):
    '''Pretty-print a matrix or vector.'''
    if len(p.shape) == 1:
        return vector_str(p, decimal_places, ignore)
    if len(p.shape) == 2:
        return matrix_str(p, decimal_places, ignore, label_columns)
    raise Exception('Invalid array with shape {0}'.format(p.shape))

def matrix_str(p, decimal_places=2, ignore=None, label_columns=False):
    '''Pretty-print the matrix.'''
    return '[{0}]'.format("\n  ".join([(str(i) if label_columns else '') + vector_str(a, decimal_places, ignore) for i, a in enumerate(p)]))

def vector_str(p, decimal_places=2, ignore=None):
    '''Pretty-print the vector values.'''
    style = '{0:.' + str(decimal_places) + 'f}'
    return '[{0}]'.format(", ".join([' ' if ((hasattr(ignore, "__len__") and a in ignore) or a == ignore) else style.format(a) for a in p]))


def batches(indices, batch_size, shuffle=True):
    order = np.copy(indices)
    if shuffle:
        np.random.shuffle(order)
    nbatches = int(np.ceil(len(order) / float(batch_size)))
    for b in range(nbatches):
        idx = order[b*batch_size:min((b+1)*batch_size, len(order))]
        yield idx


def p_value_2sided(z, mu0=0., sigma0=1.):
    return 2*(1.0 - st.norm.cdf(np.abs((z - mu0) / sigma0)))

def bh(p, fdr):
    p_orders = np.argsort(p)
    discoveries = []
    m = float(len(p_orders))
    for k, s in enumerate(p_orders):
        if p[s] <= (k+1) / m * fdr:
            discoveries.append(s)
        else:
            break
    return np.array(discoveries, dtype=int)

def bh_predictions(p_values, fdr_threshold):
    reshape = False
    if len(p_values.shape) > 1:
        reshape = True
        p_shape = p_values.shape
        p_values = p_values.flatten()
    pred = np.zeros(len(p_values), dtype=int)
    disc = bh(p_values, fdr_threshold)
    if len(disc) > 0:
        pred[disc] = 1
    if reshape:
        pred = pred.reshape(p_shape)
    return pred

def sample_gmm(pi, mu, sigma):
    p = np.random.random()
    for pi_i, mu_i, sigma_i in zip(np.cumsum(pi), mu, sigma):
        if p < pi_i:
            return np.random.normal(mu_i, sigma_i)
    return np.random.normal(mu[-1], sigma[-1])

def gmm_pdf(x, pi, mu, sigma):
    return np.array([pi_i * st.norm.pdf(x, mu_i, sigma_i) for pi_i, mu_i, sigma_i in zip(pi, mu, sigma)]).sum(axis=0)

def create_folds(X, k):
    if isinstance(X, int) or isinstance(X, np.integer):
        indices = np.arange(X)
    elif hasattr(X, '__len__'):
        indices = np.arange(len(X))
    else:
        indices = np.arange(X.shape[0])
    np.random.shuffle(indices)
    folds = []
    start = 0
    end = 0
    for f in range(k):
        start = end
        end = start + len(indices) // k + (1 if (len(indices) % k) > f else 0)
        folds.append(indices[start:end])
    return folds


def calc_fdr(probs, fdr_level):
    '''Calculates the detected signals at a specific false discovery rate given the posterior probabilities of each point.'''
    pshape = probs.shape
    if len(probs.shape) > 1:
        probs = probs.flatten()
    post_orders = np.argsort(probs)[::-1]
    avg_fdr = 0.0
    end_fdr = 0
    
    for idx in post_orders:
        test_fdr = (avg_fdr * end_fdr + (1.0 - probs[idx])) / (end_fdr + 1.0)
        if test_fdr > fdr_level:
            break
        avg_fdr = test_fdr
        end_fdr += 1

    is_finding = np.zeros(probs.shape, dtype=int)
    is_finding[post_orders[0:end_fdr]] = 1
    if len(pshape) > 1:
        is_finding = is_finding.reshape(pshape)
    return is_finding

def logsumexp(inputs, dim=None, keepdim=False):
    """Numerically stable logsumexp.

    Args:
        inputs: A Variable with any shape.
        dim: An integer.
        keepdim: A boolean.

    Returns:
        Equivalent of log(sum(exp(inputs), dim=dim, keepdim=keepdim)).

    Taken from https://github.com/pytorch/pytorch/issues/2591
    """
    # For a 1-D array x (any array along a single dimension),
    # log sum exp(x) = s + log sum exp(x - s)
    # with s = max(x) being a common choice.
    import torch
    if dim is None:
        inputs = inputs.view(-1)
        dim = 0
    s, _ = torch.max(inputs, dim=dim, keepdim=True)
    outputs = s + (inputs - s).exp().sum(dim=dim, keepdim=True).log()
    if not keepdim:
        outputs = outputs.squeeze(dim)
    return outputs

def fit_logistic(X, y, min_c=1e-10, max_c=1e4, num_c=50, num_folds=5, X_holdout=None, model='gboost', min_prob=5e-3):
    class Dummy:
        def __init__(self, const):
            self.const = const

        def predict_proba(self, X):
            p = np.ones(len(X)) * self.const
            return np.array([1-p, p]).T
    if y.sum() == 0:
        if X_holdout is None:
            return np.zeros(len(y)) + min_prob, lambda: np.zeros(len(y), dtype=int) + min_prob, Dummy(min_prob)
        else:
            return np.zeros(X_holdout.shape[0]) + min_prob, lambda: np.zeros(X_holdout.shape[0], dtype=int) + min_prob, Dummy(min_prob)
    if y.sum() == len(y):
        if X_holdout is None:
            return np.ones(len(y)) - min_prob, lambda: np.ones(len(y), dtype=int) - min_prob, Dummy(1-min_prob)
        else:
            return np.ones(X_holdout.shape[0]) - min_prob, lambda: np.ones(X_holdout.shape[0], dtype=int) - min_prob, Dummy(1-min_prob)
    if model == 'lasso':
        from sklearn.linear_model import LogisticRegression
        # Use cross-validation to select lambda
        c_vals = np.exp(np.linspace(np.log(min_c), np.log(max_c), num_c))
        cv_scores = np.zeros(num_c)
        folds = create_folds(X, num_folds)
        for i,fold in enumerate(folds):
            # print '\tFold #{0}'.format(i)
            mask = np.ones(len(X), dtype=bool)
            mask[fold] = False
            X_train, y_train = X[mask], y[mask]
            X_test, y_test = X[~mask], y[~mask]
            if y_train.sum() == 0:
                cv_scores += (1-y_test).sum()
            elif y_train.sum() == len(y_train):
                cv_scores += (y_test).sum()
            else:
                lr = LogisticRegression(penalty='l1', C=min_c, warm_start=True)
                for j,c in enumerate(c_vals):
                    lr.C = c
                    lr.fit(X_train, y_train)
                    cv_scores[j] += lr.predict_log_proba(X_test)[:,y_test].sum()
        cv_scores /= float(len(X))
        best_idx = np.argmax(cv_scores)
        best_c = c_vals[best_idx]
        lr = LogisticRegression(C=best_c)
    elif model == 'gboost':
        from sklearn.ensemble import GradientBoostingClassifier
        lr = GradientBoostingClassifier(subsample=0.5)

    lr.fit(X, y)
    if X_holdout is None:
        probs = lr.predict_proba(X)[:,1]
    else:
        probs = lr.predict_proba(X_holdout)[:,1]

    return probs, lambda: (np.random.random(size=len(probs)) <= probs).astype(int), lr
