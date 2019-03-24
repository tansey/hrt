'''
Simple example showing how a naive permutation test can lead to massively
inflated errors.
'''
import numpy as np
from collections import defaultdict
from pyhrt.utils import p_value_2sided

# Benjamini-hochberg
def bh(p, fdr):
    p_orders = np.argsort(p)
    discoveries = []
    m = float(len(p_orders))
    for k, s in enumerate(p_orders):
        if p[s] <= (k+1) / m * fdr:
            discoveries.append(s)
        else:
            break
    return np.array(discoveries)

def covariates(N, P):
    # Make a really weird distribution for X
    # Specifically, tie the X's together via
    # a latent variable Z
    Z = np.random.normal(0,3,size=(N,5))
    X = np.zeros((N,P))
    for i in range(P):
        X[:,i] = np.random.normal(size=N)
        for j in range(5):
            k = np.random.choice(3) + 1
            beta_ij = np.random.normal()
            X[:,i] += beta_ij * Z[:,j]**k
    return X

def errors(discoveries, rejections, T):
    fdp = discoveries[T:].sum() / max(1,discoveries.sum())
    fwe = np.any(rejections[T:])
    return fwe, fdp

def select(p_values, alpha, T):
    # Let's control FDR at alpha
    discoveries = bh(p_values, alpha) # Benjamini-Hochberg correction
    # Let's control Type I error rate at alpha
    rejected = (p_values < (alpha / len(p_values))) # Bonferroni correction
    return errors(discoveries, rejected, T)
    


if __name__ == '__main__':
    # 100 samples, 10 covariates, 4 non-null, repeat 100 indepent times, with error rate alpha
    N = 100
    P = 10
    T = 4
    runs = 100
    alpha = 0.05

    # reproducibility
    np.random.seed(42)

    # Quieter sklearn
    import warnings
    from sklearn.exceptions import ConvergenceWarning
    warnings.filterwarnings("ignore", category=ConvergenceWarning)

    fdp = defaultdict(lambda: np.zeros(runs))
    fwe = defaultdict(lambda: np.zeros(runs))
    ranks = defaultdict(lambda: np.zeros((runs,P)))
    for run in range(runs):
        print('Trial {}'.format(run+1))

        # Get some dependent X's that are not simply MVN
        X = covariates(N, P)

        # Only the first T covariates actually non-null
        beta = np.random.normal(size=T)

        # The true y is some nonlinear function but it looks pretty good when you
        # try approximating it with a linear model.
        y = np.tanh(beta[0]*X[:,0] + beta[1] * X[:,3]) * np.tanh(beta[2]*X[:,2] + beta[3]*X[:,3]) + np.random.normal(size=N)

        # Fit beta_hat using least squares
        beta_hat = np.linalg.inv(X.T.dot(X)).dot(X.T.dot(y))

        ''' Use normal assumption for beta '''
        p_values = p_value_2sided(beta_hat)
        fwe['normal'][run], fdp['normal'][run] = select(p_values, alpha, T)

        ''' Let's test how likely you would see an effect that big under a randomization null for each feature'''
        ntrials = 1000
        p_values = np.zeros(P)
        for j in range(P):
            nulls = np.zeros(ntrials)
            X_null = np.copy(X)
            indices = np.arange(N)
            for trial in range(ntrials):
                # Randomize the j'th column
                np.random.shuffle(indices)
                X_null[:,j] = X_null[indices,j]

                # Refit the model
                beta_null = np.linalg.inv(X_null.T.dot(X_null)).dot(X_null.T.dot(y))

                # How big is the coefficient?
                nulls[trial] = np.abs(beta_null[j])

            # Permutation test 
            p_values[j] = (nulls >= np.abs(beta_hat[j])).mean()
        
        fwe['perm'][run], fdp['perm'][run] = select(p_values, alpha, T)

        ''' Fit a Bayesian linear model with ARD '''
        from scipy.stats import norm
        from sklearn.linear_model import ARDRegression
        ard = ARDRegression()
        ard.fit(X, y)
        cdfs = np.array([norm.cdf(0, mu, sigma) for mu, sigma in zip(ard.coef_,np.sqrt(np.diag(ard.sigma_)))])
        discoveries = (cdfs < alpha / 2) | (cdfs > (1-alpha/2))
        fdp['bayes'][run] = discoveries[T:].sum() / max(1,discoveries.sum())

        ''' Fit a lasso model '''
        from sklearn.linear_model import LassoCV
        lasso = LassoCV()
        lasso.fit(X, y)
        fwe['lasso'][run], fdp['lasso'][run] = errors(np.abs(lasso.coef_) > 1e-4, np.abs(lasso.coef_) > 1e-4, T)
        # Lasso top 4
        ranks['lasso'][run][np.argsort(np.abs(lasso.coef_))[::-1]] = np.arange(P)
        rank_select = (ranks['lasso'][run] < T) & (np.abs(lasso.coef_) > 1e-4)
        fwe['lasso_ranks'], fdp['lasso_ranks'] = errors(rank_select, rank_select, T)
        ranks['lasso'][run][np.abs(lasso.coef_) < 1e-4] = np.nan

        ''' Fit a random forest model '''
        from sklearn.ensemble import RandomForestRegressor
        rf = RandomForestRegressor(n_estimators=100, n_jobs=8)
        rf.fit(X,y)
        ranks['rf'][run][np.argsort(rf.feature_importances_)[::-1]] = np.arange(P)
        rank_select = ranks['rf'][run] < T
        fwe['rf_ranks'], fdp['rf_ranks'] = errors(rank_select, rank_select, T)
    
    print('Bayesian ridge:')
    print('Estimated FDR: {:.2f}'.format(fdp['bayes'].mean()*100)) # 67.77%    
    print('')
    print('Lasso:')
    print('Estimated Type I error rate: {:.2f}'.format(fwe['lasso'].mean()*100)) # 98.00%
    print('Estimated FDR: {:.2f}'.format(fdp['lasso'].mean()*100)) # 67.77%    
    print('')
    print('Lasso ranks:')
    print('Estimated Type I error rate: {:.2f}'.format(fwe['lasso_ranks'].mean()*100)) # 98.00%
    print('Estimated FDR: {:.2f}'.format(fdp['lasso_ranks'].mean()*100)) # 67.77%
    for feat, rank in enumerate(np.nanmean(ranks['lasso'],axis=0)):
        print('{}. {}'.format(feat, rank))
    print('')
    print('Estimated Type I error rate: {:.2f}'.format(fwe['rf_ranks'].mean()*100)) # 98.00%
    print('Estimated FDR: {:.2f}'.format(fdp['rf_ranks'].mean()*100)) # 67.77%
    print('Random forest ranks:')
    for feat, rank in enumerate(ranks['rf'].mean(axis=0)):
        print('{}. {}'.format(feat, rank))


