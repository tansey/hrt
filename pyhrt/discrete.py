import os
import sys
import numpy as np
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.optim as optim
from scipy.stats.mstats import gmean
from pyhrt.utils import batches, create_folds, ilogit

############################################################
'''Discrete conditionals'''
############################################################
class MultinomialModel:
    def __init__(self, probs, classes=None):
        self.probs = probs
        self.classes = classes if classes is not None else np.arange(probs.shape[-1])

    def sample(self):
        return self.classes[np.array([np.random.choice(self.probs.shape[1], p=p) for p in self.probs])]

    def pmf(self, y):
        if hasattr(y, '__len__'):
            assert len(y) == len(self.probs)
            if len(y.shape) == 2:
                assert (y[:,:,None] == self.classes[None,None,:]).max(axis=2).min() == 1
                return np.array([p[np.argmax(y_i[:,None]==self.classes[None,:], axis=0)] for y_i, p in zip(y, self.probs)])
            return np.array([p[np.argmax(y_i==self.classes)] for y_i, p in zip(y, self.probs)])
        return self.probs[:,np.argmax(y==self.classes)]

    def cmf(self, y):
        if hasattr(y, '__len__'):
            assert len(y) == len(self.probs)
            return np.array([p[:np.argmax(y_i==self.classes)+1].sum() for y_i, p in zip(y, self.probs)])
        return self.probs[:,:self.argmax(y==self.classes)+1].sum(axis=1)

    def prob(self, y):
        return self.pmf(y)


'''Neural discrete probabilistic classifier'''
class DiscreteClassifierNetwork(nn.Module):
    def __init__(self, nfeatures, classes, X_means, X_stds):
        super(DiscreteClassifierNetwork, self).__init__()
        self.classes = classes
        self.nclasses = len(classes)
        self.X_means = X_means
        self.X_stds = X_stds
        self.fc_in = nn.Sequential(
                nn.Linear(nfeatures, 200),
                nn.ReLU(),
                nn.Dropout(),
                nn.Linear(200, 200),
                nn.ReLU(),
                nn.Dropout(),
                nn.Linear(200, self.nclasses))
        # self.fc_in = nn.Sequential(nn.Linear(nfeatures,nclasses))
        self.softmax = nn.Softmax(dim=1)
    
    def forward(self, x):
        return self.fc_in(x)

    def predict(self, X):
        self.eval()
        self.zero_grad()
        tX = autograd.Variable(torch.FloatTensor((X - self.X_means[np.newaxis,:]) / self.X_stds[np.newaxis,:]), requires_grad=False)
        logits = self.forward(tX)
        probs = self.softmax(logits)
        return MultinomialModel(probs.data.numpy(), self.classes)


'''Bootstrap confidence interval probability estimator'''
class DiscreteBootstrapConditionalModel:
    def __init__(self, X, y, fit_fn, nbootstraps=100):
        self.indices = [np.random.choice(np.arange(X.shape[0]), replace=True, size=X.shape[0]) for _ in range(nbootstraps)]
        self.models = []
        for i,idx in enumerate(self.indices):
            print('\tBootstrap {}'.format(i))
            self.models.append(fit_fn(X[idx], y[idx]))

    def pmf_quantiles(self, X, y, q, axis=None):
        return np.percentile(np.array([m.predict(X).pmf(y) for m in self.models]), q, axis=axis)

    def cmf_quantiles(self, X, y, q, axis=None):
        return np.percentile(np.array([m.predict(X).cmf(y) for m in self.models]), q, axis=axis)

    def sample(self, X):
        return self.models[0].predict(X, return_std=True).sample()


def fit_classifier(X, y, classes=None,
                  nepochs=40, val_pct=0.1,
                  batch_size=None, target_batch_pct=0.01,
                  min_batch_size=10, max_batch_size=100,
                  verbose=False, lr=3e-4, weight_decay=5e-5):
    if classes is None:
        classes = np.unique(y)
    nclasses = classes.shape[0]

    # Create a temporary file to store the best method
    import uuid
    tmp_file = '/tmp/tmp_file_' + str(uuid.uuid4())

    # Choose a suitable batch size
    if batch_size is None:
        batch_size = max(min_batch_size, min(max_batch_size, int(np.round(X.shape[0]*target_batch_pct))))

    # Standardize the features (helps with gradient propagation)
    Xstd = X.std(axis=0)
    Xstd[Xstd == 0] = 1 # Handle constant features
    tX = autograd.Variable(torch.FloatTensor((X - X.mean(axis=0,keepdims=True)) / Xstd[np.newaxis, :]), requires_grad=False)

    # Create the classes using their indices
    tY = np.zeros(y.shape[0], dtype=int)
    for i,c in enumerate(classes):
        tY[y == c] = i
    tY = autograd.Variable(torch.LongTensor(tY), requires_grad=False)

    # Training weights to balance the dataset
    y_counts = np.array([(y == c).sum() for c in classes])
    tY_weights = autograd.Variable(torch.FloatTensor(len(y_counts) * y_counts / float(len(y))), requires_grad=False)
    crossent = nn.CrossEntropyLoss(weight=tY_weights)

    # Create train/validate splits
    indices = np.arange(X.shape[0], dtype=int)
    np.random.shuffle(indices)
    train_cutoff = int(np.round(len(indices)*(1-val_pct)))
    train_indices = indices[:train_cutoff]
    validate_indices = indices[train_cutoff:]

    model = DiscreteClassifierNetwork(X.shape[1], classes, X.mean(axis=0), Xstd)

    # Setup the SGD method
    optimizer = optim.RMSprop(model.parameters(), lr=lr, weight_decay=weight_decay)

    # Track progress
    train_losses, val_losses, best_loss = np.zeros(nepochs), np.zeros(nepochs), None
    
    # Train the model
    for epoch in range(nepochs):
        if verbose:
            print('\t\tEpoch {}'.format(epoch+1))
            sys.stdout.flush()

        # Track the loss curves
        train_loss = torch.Tensor([0])
        for batch_idx, batch in enumerate(batches(train_indices, batch_size, shuffle=True)):
            if verbose and (batch_idx % 100 == 0):
                print('\t\t\tBatch {}'.format(batch_idx))
            tidx = autograd.Variable(torch.LongTensor(batch), requires_grad=False)

            # Set the model to training mode
            model.train()

            # Reset the gradient
            model.zero_grad()

            # Run the model and get the predictions
            logits = model(tX[tidx])

            # Cross-entropy loss
            loss = crossent(logits, tY[tidx])

            # Calculate gradients
            loss.backward()

            # Apply the update
            # [p for p in model.parameters() if p.requires_grad]
            optimizer.step()

            # Track the loss
            train_loss += loss.data

        validate_loss = torch.Tensor([0])
        for batch_idx, batch in enumerate(batches(validate_indices, batch_size, shuffle=False)):
            if verbose and (batch_idx % 100 == 0):
                print('\t\t\tValidation Batch {}'.format(batch_idx))
            tidx = autograd.Variable(torch.LongTensor(batch), requires_grad=False)

            # Set the model to test mode
            model.eval()

            # Reset the gradient
            model.zero_grad()

            # Run the model and get the predictions
            logits = model(tX[tidx])

            # Cross-entropy loss
            loss = crossent(logits, tY[tidx])

            # Track the loss
            validate_loss += loss.data

        train_losses[epoch] = train_loss.numpy() / float(len(train_indices))
        val_losses[epoch] = validate_loss.numpy() / float(len(validate_indices))

        # Check if we are currently have the best held-out log-likelihood
        if epoch == 0 or val_losses[epoch] <= best_loss:
            if verbose:
                print('\t\t\tSaving test set results.      <----- New high water mark on epoch {}'.format(epoch+1))
            # If so, use the current model on the test set
            best_loss = val_losses[epoch]
            torch.save(model, tmp_file)
        
        if verbose:
            print('Validation loss: {} Best: {}'.format(val_losses[epoch], best_loss))

    model = torch.load(tmp_file)
    os.remove(tmp_file)
    return model

def pb_test(tstat, probs, ntrials=10000):
    '''Discrete goodness of fit test using propensity balancing. TODO: did not use this; need to rethink it.'''
    samples = np.array([np.random.choice(len(p), p=p, size=ntrials, replace=True) for p in probs]).T
    weights = 1. / np.array([p[s] for p, s in zip(probs, samples.T)]).T
    t_null = np.zeros(ntrials)
    for tidx in range(ntrials):
        w = weights[tidx]
        s = samples[tidx]
        t_null[tidx] = np.abs(probs.shape[0] - np.array([w[s == i].sum() for i in range(probs.shape[1])])).sum()
    return (tstat >= t_null).mean()

def tv_test(tvstat, marginals, nsamples, ntrials=10000):
    samples = np.random.choice(len(marginals), p=marginals, size=(ntrials,nsamples), replace=True)
    sample_marginals = np.array([(samples == i).mean(axis=1) for i in range(len(marginals))]).T
    sample_tv = np.abs(marginals[np.newaxis,:] - sample_marginals).sum(axis=1)
    return (tvstat >= sample_tv).mean()

def sample_cv(X, models, folds, lower, upper):
    y = np.zeros(X.shape[0])
    probs = np.zeros((X.shape[0],3))
    for model, fold in zip(models, folds):
        y[fold], probs[fold] = sample_holdout(X[fold], model, lower, upper)
    return y, probs

def sample_holdout(X, model, lower, upper):
    y = model.models[0].predict(X).sample()
    probs = np.zeros((X.shape[0], 3))
    probs[:,0] = model.models[0].predict(X).pmf(y)
    probs[:,1] = model.pmf_quantiles(X, y, lower, axis=0)
    probs[:,2] = model.pmf_quantiles(X, y, upper, axis=0)
    return y, probs

def sample_holdout_dists(dists, model, quantiles):
    y = dists[0].sample()
    logpdfs = np.log(np.array([d.pmf(y) for d in dists]).clip(1e-100, np.inf))
    if quantiles is None:
        return y, None
    probs = np.exp(logpdfs - logpdfs[0:1]) # likelihood ratio
    quants = np.percentile(probs, quantiles, axis=0) # quantile per-sample
    quants = gmean(quants, axis=1) # (geometric) mean quantile
    return y, quants

class CrossValidationSampler:
    def __init__(self, X, models, folds, quantiles=None):
        self.N = X.shape[0]
        self.models = models
        self.folds = folds
        self.quantiles = quantiles
        self.dists = [[m.predict(X[fold]) for m in model_set.models] for model_set, fold in zip(self.models, self.folds)]

    def __call__(self):
        y = np.zeros(self.N)
        probs = np.zeros(self.N)
        if self.quantiles is not None:
            quants = np.zeros((self.N, len(self.quantiles)))
        for model, fold, dist in zip(self.models, self.folds, self.dists):
            y[fold], q = sample_holdout_dists(dist, model, self.quantiles)
            if q is not None:
                quants[fold] = q
        return y, quants

class HoldoutSampler:
    def __init__(self, X, model, quantiles=None):
        self.model = model
        self.quantiles = quantiles
        self.dists = [m.predict(X) for m in model.models]

    def __call__(self):
        return sample_holdout_dists(self.dists, self.model, self.quantiles)

# TODO: Refactor this to use classes for samplers; generalize code to remove redundancies
def calibrate_discrete(X, feature,
                       X_test=None, nquantiles=101, nbootstraps=100,
                       nfolds=5, tv_threshold=0.005, p_threshold=0.,
                       use_cv=False):
    '''Calibrates a bootstrap confidence interval conditional model for a given feature.'''
    classes = np.unique(X[:,feature])
    nclasses = len(classes)

    # Search over a linear quantile grid to search
    quantile_range = np.linspace(0, 100, nquantiles)

    jmask = np.ones(X.shape[1], dtype=bool)
    jmask[feature] = False
    if X_test is None and use_cv:
        # Use k-fold cross-validation to generate conditional probability estimates for X_j
        print('Fitting using {} bootstrap resamples and {} folds'.format(nbootstraps, nfolds))
        probs = np.zeros((nquantiles, X.shape[0], nclasses))
        proposals = []
        folds = create_folds(X, nfolds)
        for fold_idx, fold in enumerate(folds):
            print(fold_idx)
            imask = np.ones(X.shape[0], dtype=bool)
            imask[fold] = False
            model = DiscreteBootstrapConditionalModel(X[imask][:,jmask], X[imask][:,feature], fit_classifier, nbootstraps=nbootstraps)
            # probs[:,fold] = model.pmf_quantiles(X[fold][:,jmask], X[fold][:,feature], quantile_range, axis=0)
            for c in classes:
                probs[:,fold,c] = model.pmf_quantiles(X[fold][:,jmask], c, quantile_range, axis=0)
            proposals.append(model)
        # sampler = lambda l, u: sample_cv(X[:,jmask], proposals, folds, l, u)
        sampler = CrossValidationSampler(X[:,jmask], proposals, folds)
        outcomes = np.array([(X[:,feature] == c).mean() for c in classes])
    else:
        if X_test is None:
            print('Using training set as testing set.')
            X_test = X
        # Use a held-out test set
        print('Fitting using {} bootstrap resamples and a {}/{} train/test split'.format(nbootstraps, X.shape[0], X_test.shape[0]))
        model = DiscreteBootstrapConditionalModel(X[:,jmask], X[:,feature], fit_classifier, nbootstraps=nbootstraps)
        probs = np.zeros((nquantiles, X_test.shape[0], nclasses))
        for cidx,c in enumerate(classes):
            probs[:,:,cidx] = model.pmf_quantiles(X_test[:,jmask], c, quantile_range, axis=0)
        # sampler = lambda l, u: sample_holdout(X_test[:,jmask], model, l, u)
        sampler = HoldoutSampler(X_test[:,jmask], model)
        outcomes = np.array([(X_test[:,feature] == c).mean() for c in classes])

    # Find the lower quantile that forms a sufficient lower bound on the observed probabilities
    for i in range(1,nquantiles//2):
        lower = quantile_range[nquantiles//2 - i]
        qlower = probs[nquantiles//2 - i]
        tv_lower = (qlower.mean(axis=0) - outcomes).clip(0,np.inf).sum()
        tv_pvalue = tv_test(tv_lower, outcomes, probs.shape[1])
        # print('Lower: {} TV: {} p: {}'.format(lower, tv_lower, tv_pvalue))

        # Allow some error tolerance due to noise/finite data
        if tv_lower <= tv_threshold or tv_pvalue <= p_threshold:
            break

    # Find the upper quantile
    for i in range(1,nquantiles//2):
        upper = quantile_range[nquantiles//2+i]
        qupper = probs[nquantiles//2 + i]
        tv_upper = (outcomes - qupper.mean(axis=0)).clip(0,np.inf).sum()

        tv_pvalue = tv_test(tv_upper, outcomes, probs.shape[1])
        # print('Upper: {} TV: {} p: {}'.format(upper, tv_upper, tv_pvalue))
        
        # Allow some error tolerance due to noise/finite data
        if tv_upper <= tv_threshold or tv_pvalue <= p_threshold:
            break
    
    # Our TV-distance is the worst-case of the two bounds
    tv_stat = np.max([tv_lower, tv_upper])
    sampler.quantiles = np.array([lower, upper])

    # TODO: how do we get a p-value estimate here? no clear notion of null...

    print('Selected intervals: [{},{}]'.format(lower, upper))

    return {'model': model,
            'probs': probs,
            'tv_stat': tv_stat,
            'upper': upper,
            'lower': lower,
            'qupper': qupper,
            'qlower': qlower,
            'quantiles': quantile_range,
            'sampler': sampler
            }


def test_classifier():
    # Generate the ground truth
    N = 1000
    X = np.random.normal(size=(N,2))
    true_logits = (np.array([0.5,1,1.5])[np.newaxis,:]*X[:,0:1]
                    + np.array([-2,1,-0.5])[np.newaxis,:]*X[:,1:2]
                    + X[:,0:1] * X[:,1:2] * np.array([-2,1,2])[np.newaxis,:])
    truth = np.exp(true_logits) / np.exp(true_logits).sum(axis=1, keepdims=True)
    true_model = MultinomialModel(truth)
    
    # Sample some observations
    y = true_model.sample()

    # import matplotlib.pylab as plt
    # fig, axarr = plt.subplots(1,3)
    # x1, x2 = np.meshgrid(np.linspace(-3,3,100), np.linspace(-3,3,100))
    # im = np.zeros((3,100,100))
    # for c in range(3):
    #     for i in range(100):
    #         for j in range(100):
    #             b1 = [0.5, 1, 1.5]
    #             b2 = [-2, 1, -0.5]
    #             b3 = [-2, 1, 2]
    #             im[c,i,j] = np.exp(b1[c]*x1[i,j] + b2[c]*x2[i,j] - b3[c]*x1[i,j]*x2[i,j])
    # for c in range(3):
    #     gim = axarr[c].imshow(im[c] / im.sum(axis=0), vmin=0, vmax=1, interpolation='none')
    #     axarr[c].set_xlabel('X1')
    #     axarr[c].set_ylabel('X2')
    #     axarr[c].set_title('Prob(c={})'.format(c))
    # fig.subplots_adjust(right=0.8)
    # cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
    # fig.colorbar(gim, cax=cbar_ax)
    # plt.show()

    # Fit the model
    split = int(np.round(N)*0.8)
    model = fit_classifier(X[:split], y[:split], verbose=True)
    
    # Predict the likelihood of observations
    pred = model.predict(X[split:]).probs

    import matplotlib.pylab as plt
    plt.scatter(truth[split:], pred, color='blue')
    plt.plot([0,1],[0,1], color='red', lw=3)
    plt.xlabel('Truth')
    plt.ylabel('Predicted')
    plt.show()

def test_calibration():
    # Generate the ground truth
    N = 1000
    X = np.random.normal(size=(N,2))
    true_logits = (np.array([0.5,1,1.5])[np.newaxis,:]*X[:,0:1]
                    + np.array([-2,1,-0.5])[np.newaxis,:]*X[:,1:2]
                    + X[:,0:1] * X[:,1:2] * np.array([-2,1,2])[np.newaxis,:])
    truth = np.exp(true_logits) / np.exp(true_logits).sum(axis=1, keepdims=True)
    true_model = MultinomialModel(truth)
        
    # Sample some observations
    y = true_model.sample()
    Xy = np.concatenate([X,y[:,np.newaxis]], 1)

    # Fit the calibrated model
    split = int(np.round(X.shape[0]*0.8))
    results = calibrate_discrete(Xy[:split], 2, X_test=Xy[split:], nbootstraps=100)
    print(results)

    # look at the bounds
    (model, probs,
     tv_stat,
     upper, lower,
     qupper, qlower,
     quantile_range) = (results['model'],
                        results['probs'],
                        results['tv_stat'],
                        results['upper'],
                        results['lower'],
                        results['qupper'], 
                        results['qlower'],
                        results['quantiles'])
    print('Quantile chosen: [{},{}] TV={}'.format(lower, upper, tv_stat))

    plt.clf()
    tpoints = np.array([ti[yi] for yi, ti in zip(y,truth[split:])])
    lpoints = np.array([qi[yi] for yi, qi in zip(y, qlower)])
    upoints = np.array([qi[yi] for yi, qi in zip(y, qupper)])
    plt.scatter(tpoints, lpoints, color='orange', label='{:.0f}% quantile'.format(lower))
    plt.scatter(tpoints, upoints, color='blue', label='{:.0f}% quantile'.format(upper))
    for t,l,u in zip(tpoints, lpoints, upoints):
        plt.plot([t,t],[l,u], color='gray', alpha=0.5)
    plt.plot([0,1],[0,1], color='red')
    plt.xlabel('Truth')
    plt.ylabel('Estimated')
    plt.legend(loc='upper left')
    plt.savefig('plots/discrete-calibration-scatter.pdf', bbox_inches='tight')






