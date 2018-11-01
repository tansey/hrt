# pyhrt -- A Package for Holdout Randomization Tests (HRTs)

HRTs enable you to perform feature selection with statistical guarantees. Specifically, if you fit any predictive model, HRTs can use its performance on the test data under various randomizations to assign a valid (or conservative) p-value to each feature. The p-value corresponds to an estimate of the outcome of a conditional independence test, testing if feature `X_j` is independent of the response `y` given the remaining variables `X_{-j}`.

## Installation
Clone the repo and run `python setup.py install`

## Running an HRT

Coming soon... see `examples/olfaction/main.py` until then.

## Citing HRTs and pyHRT
