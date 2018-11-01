import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import os
from pyhrt.utils import create_folds, pretty_str
from pyhrt.hrt import hrt
from pyhrt.continuous import calibrate_continuous
from pyhrt.discrete import calibrate_discrete

# Run elastic net regression for each drug, choosing lambda via 10-fold CV
def fit_elastic_net_ccle(X, y, nfolds=10):
    from sklearn.linear_model import ElasticNetCV
    # The parameter l1_ratio corresponds to alpha in the glmnet R package 
    # while alpha corresponds to the lambda parameter in glmnet
    # enet = ElasticNetCV(l1_ratio=np.linspace(0.2, 1.0, 10),
    #                     alphas=np.exp(np.linspace(-6, 5, 250)),
    #                     cv=nfolds)
    enet = ElasticNetCV(l1_ratio=0.2, # It always chooses l1_ratio=0.2
                        alphas=np.exp(np.linspace(-6, 5, 250)),
                        cv=nfolds)
    print('Fitting via CV')
    enet.fit(X,y)
    alpha, l1_ratio = enet.alpha_, enet.l1_ratio_
    print('Chose values: alpha={}, l1_ratio={}'.format(alpha, l1_ratio))
    return enet

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

def plot_ccle_predictions(model, X, y):
    from sklearn.metrics import r2_score
    plt.close()
    y_hat = model.predict(X)
    plt.scatter(y_hat, y, color='blue')
    plt.plot([min(y.min(), y_hat.min()),max(y.max(), y_hat.max())], [min(y.min(), y_hat.min()),max(y.max(), y_hat.max())], color='red', lw=3)
    plt.xlabel('Predicted')
    plt.ylabel('Truth')
    plt.title('{} ($r^2$={:.4f})'.format(model.name, r2_score(y, y_hat)))
    plt.tight_layout()
    plt.savefig('plots/ccle-predictions.pdf', bbox_inches='tight')
    plt.close()

def load_ccle(feature_type = 'expression', drug_target=None, normalize=False):
    if feature_type in ['expression', 'both']:
        # Load gene expression
        expression = pd.read_csv('data/expression.txt', delimiter='\t', header=2, index_col=1).iloc[:,1:]
        expression.columns = [c.split(' (ACH')[0] for c in expression.columns]
        features = expression
    if feature_type in ['mutation', 'both']:
        # Load gene mutation
        mutations = pd.read_csv('data/mutation.txt', delimiter='\t', header=2, index_col=1).iloc[:,1:]
        mutations = mutations.iloc[[c.endswith('_MUT') for c in mutations.index]]
        features = mutations
    if feature_type == 'both':
        both_cells = set(expression.columns) & set(mutations.columns)
        z = {}
        for c in both_cells:
            exp = expression[c].values
            if len(exp.shape) > 1:
                exp = exp[:,0]
            z[c] = np.concatenate([exp, mutations[c].values])
        both_df = pd.DataFrame(z, index=[c for c in expression.index] + [c for c in mutations.index])
        features = both_df
    response = pd.read_csv('data/response.csv', header=0, index_col=[0,2])
    # Get per-drug X and y regression targets
    cells = response.index.levels[0]
    drugs = response.index.levels[1]
    X_drugs = [[] for _ in drugs]
    y_drugs = [[] for _ in drugs]
    for j,drug in enumerate(drugs):
        if drug_target is not None and drug != drug_target:
            continue
        for i,cell in enumerate(cells):
            if cell not in features.columns or (cell, drug) not in response.index:
                continue
            X_drugs[j].append(features[cell].values)
            y_drugs[j].append(response.loc[(cell,drug), 'Amax'])
        print('{}: {}'.format(drug, len(y_drugs[j])))
    X_drugs = [np.array(x_i) for x_i in X_drugs]
    y_drugs = [np.array(y_i) for y_i in y_drugs]
    if normalize:
        X_drugs = [(x_i if (len(x_i) == 0) else (x_i - x_i.mean(axis=0,keepdims=True)) / x_i.std(axis=0).clip(1e-6)) for x_i in X_drugs]
        y_drugs = [(y_i if (len(y_i) == 0 or y_i.std() == 0) else (y_i - y_i.mean()) / y_i.std()) for y_i in y_drugs]
    return X_drugs, y_drugs, drugs, cells, features.index

def ccle_feature_filter(X, y, threshold=0.1):
    # Remove all features that do not have at least pearson correlation at threshold with y
    corrs = np.array([np.abs(np.corrcoef(x, y)[0,1]) if x.std() > 0 else 0 for x in X.T])
    selected = corrs >= threshold
    print(selected.sum(), selected.shape, corrs[34758])
    return selected, corrs

def print_top_features(model, expected):
    model_weights = np.mean([m.coef_ for m in model.models], axis=0)
    print('CCLE selected:')
    for gene_target in expected:
        print('{}: {}'.format(gene_target, model_weights[ccle_features.get_loc(gene_target)]))
    print('')
    print('Top by fit:')
    for idx, top in enumerate(np.argsort(np.abs(model_weights))[::-1][:10]):
        print('{}. {}: {:.4f}'.format(idx+1, ccle_features[top], model_weights[top]))

def load_plx4720(verbose=False):
    drug_target = 'PLX4720'
    if verbose:
        print('Loading data')
    X_drugs, y_drugs, drugs, cells, features = load_ccle(feature_type='both', drug_target=drug_target, normalize=True)
    drug_idx = drugs.get_loc(drug_target)
    if verbose:
        print('Drug {}'.format(drugs[drug_idx]))
    X_drug, y_drug = X_drugs[drug_idx], y_drugs[drug_idx]

    ######## Specific to PLX4720. Filters out all features with pearson correlation less than 0.1 in magnitude ########
    ccle_expected_features = ['C11orf85', 'FXYD4', 'SLC28A2', 'MAML3_MUT', 'RAD51L1_MUT', 'GAPDHS', 'BRAF_MUT']
    if verbose:
        print('Filtering by correlation with signal first')
    ccle_selected, corrs = ccle_feature_filter(X_drug, y_drug)
    for plx4720_feat in ccle_expected_features:
        idx = features.get_loc(plx4720_feat)
        ccle_selected[idx] = True
        if verbose:
            print('Correlation for {}: {:.4f}'.format(plx4720_feat, corrs[idx]))
    ccle_features = features[ccle_selected]

    # Split the data into 10 folds where each fold contains at least 1 observation of each drug
    nfolds = 10
    drug_folds = [create_folds(x_i, nfolds) for x_i in X_drugs]
    X_drug, y_drug, folds = X_drugs[drug_idx], y_drugs[drug_idx], drug_folds[drug_idx]

    # Load or fit the model
    MODEL_PATH = 'data/model.pt'
    if os.path.exists(MODEL_PATH):
        elastic_model = torch.load(MODEL_PATH)
    else:
        elastic_model = CvModel(fit_cv(X_drug, y_drug, folds, fit_elastic_net_ccle, selected=ccle_selected), folds, 'Elastic Net', selected=ccle_selected)
        torch.save(elastic_model, MODEL_PATH)

        # Plot the fit to show it's pretty good
        plot_ccle_predictions(elastic_model, X_drug, y_drug)

        # Show the features selected by the heuristic if we fit this way (may be slightly different than in the paper)
        if verbose:
            print_top_features(elastic_model, ccle_expected_features)

    return X_drug, y_drug, features, ccle_features, elastic_model

def get_model_weights(elastic_model):
    return np.mean([m.coef_ for m in elastic_model.models], axis=0)

def run_hrt(feat_idx, X_drug, y_drug, elastic_model,
            features, ccle_features,
            pca_components=100, discrete_threshold=10,
            nbootstraps=100, nperms=5000, verbose=False):
    gene_target = ccle_features[feat_idx]
    feature = features.get_loc(gene_target)
    nunique = np.unique(X_drug[:,feature]).shape[0]
    if verbose:
        print('{} is feature number {} with {} unique values'.format(gene_target, feature, nunique))
    fmask = np.ones(X_drug.shape[1], dtype=bool)
    fmask[feature] = False
    X_transform = X_drug[:,fmask]
    from sklearn.decomposition import PCA
    pca = PCA(n_components=pca_components)
    X_transform = pca.fit_transform(X_transform)
    X_transform = np.concatenate([X_drug[:,feature:feature+1], X_transform], axis=1)
    if nunique <= discrete_threshold:
        if verbose:
            print('Using discrete conditional')
        results = calibrate_discrete(X_transform, 0, nbootstraps=nbootstraps)
    else:
        if verbose:
            print('Using continuous conditional')
        results = calibrate_continuous(X_transform, 0, nbootstraps=nbootstraps)
    conditional = results['sampler']
    tstat = lambda X_test: ((y_drug - elastic_model.predict(X_test))**2).mean()
    p_value = hrt(feature, tstat, X_drug, nperms=nperms,
                        conditional=conditional,
                        lower=conditional.quantiles[0],
                        upper=conditional.quantiles[1])['p_value']
    return p_value

if __name__ == '__main__':
    np.random.seed(42)
    torch.manual_seed(42)
    torch.set_num_threads(1)
    
    X_drug, y_drug, features, ccle_features, elastic_model = load_plx4720()

    # Get the weights to use in feature ranking
    model_weights = get_model_weights(elastic_model)

    # Run the HRT for the top 10 features sorted by average coefficient magnitude
    ntop = 10
    p_values = np.zeros(ntop)
    top_features = np.argsort(np.abs(model_weights))[::-1][:ntop]
    for rank, feat_idx in enumerate(top_features):
        gene_target = ccle_features[feat_idx]
        p_value = run_hrt(feat_idx, X_drug, y_drug, elastic_model, features, ccle_features, verbose=True)
        print('{}. {} p={}'.format(rank+1, gene_target, p_value))
        p_values[rank] = p_value

    print('p-values for {}:'.format(drug_target))
    for rank, feat_idx in enumerate(top_features):
        gene_target = ccle_features[feat_idx]
        print('{}. {}:\t\tweight={:.4f} p={:.4f}'.format(rank+1, gene_target, model_weights[top], p_values[rank]))

'''
p-values for PLX4720:
1. BRAF.V600E_MUT:      weight=-0.0993 p=0.0007
2. RP11-208G20.3:       weight=-0.0708 p=0.0073
3. RP6-149D17.1:        weight=-0.0684 p=0.0233
4. RNA5SP184:       weight=-0.0638 p=0.0202
5. RNU6-104P:       weight=-0.0612 p=0.6259
6. MTMR11_MUT:      weight=-0.0564 p=0.0869
7. RP11-567M16.3:       weight=0.0546 p=0.0794
8. VPS13B_MUT:      weight=0.0544 p=0.0157
9. RP1-167G20.1:        weight=0.0511 p=0.6449
10. HIP1_MUT:       weight=-0.0503 p=0.3237
'''

