# pyhrt -- A Package for Holdout Randomization Tests (HRTs)

HRTs enable you to perform feature selection with statistical guarantees. Specifically, if you fit any predictive model, HRTs can use its performance on the test data under various randomizations to assign a valid (or conservative) p-value to each feature. The p-value corresponds to an estimate of the outcome of a conditional independence test, testing if feature `X_j` is independent of the response `y` given the remaining variables `X_{-j}`.

## Installation
Clone the repo and run `python setup.py install`

## Running an HRT

To run an HRT for a specific feature, define the following in your code:

- `feature`: The index (column) of the feature in a numpy array.

- `tstat_fn`: The test statistic used to evaluate a predictive model. This should be a function or callable that takes `X_test`, a test array of perturbed `X` values, and returns a score for the predictive accuracy of your model on a test `Y` using `X_test`. 

- `X`: A numpy array of `samples` by `features`. By default, we assume that this is the test set of `X`s on which to evaluate the model.

## Simple example

```
n = 1000
p = 30
X = np.random.normal(size=(n,1)) * 0.5 + np.random.normal(size=(n,p)) * 0.5
beta = np.random.normal(size=4)
Y = np.random.normal(X[:,1:1+len(beta)].dot(beta))

# Create train/test splits
X_train, Y_train = X[:900], Y[:900]
X_test, Y_test = X[900:], Y[900:]

# Simple OLS regression
from sklearn.linear_model import LinearRegression
model = LinearRegression().fit(X_train, Y_train)

# Evaluate the first feature
feature = 0

# Use mean squared error as the empirical risk metric
tstat_fn = lambda X_eval: ((Y_test - model.predict(X_eval))**2).mean()

# Run the HRT
from pyhrt.hrt import hrt
results = hrt(feature, tstat_fn, X_test)

# Print the p-value
print(results['p_value'])
```


## Examples using the HRT

Three examples are available. 

1) See `examples/olfaction/` for the case study looking at olfactory responses to various compounds (Science 2017).

2) See `examples/ccle/` for the case study looking at dose-response modeling in cancer cell lines (Nature 2012).

3) See `examples/timings/` for a simple set of experiments with all of the algorithms implemented in the paper. This contains basic implementations that are useful for understanding the core of the HRT and coding your own variations and implementations. All methods assume access to the true complete conditionals.

## Citing HRTs and pyHRT
```
The Holdout Randomization Test for Feature Selection in Black Box Models
W. Tansey, V. Veitch, H. Zhang, R. Rabadan, and D. M. Blei
Journal of Computational and Graphical Statistics, 2021.
```