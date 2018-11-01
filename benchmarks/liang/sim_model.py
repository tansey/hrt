import os
import numpy as np
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.optim as optim
from pyhrt.utils import create_folds, batches

class PermutationConditional:
    def __init__(self, X, j, folds=None):
        self.X = X[:,j]
        self.folds = folds

    def __call__(self):
        if self.folds is None:
            result = np.copy(self.X)
            np.random.shuffle(result)
        else:
            result = np.zeros(self.X.shape[0])
            for fold in self.folds:
                fold_x = np.array(self.X[fold])
                np.random.shuffle(fold_x)
                result[fold] = fold_x
        return result, np.ones(2)

'''Use a simple neural regression model'''
class NeuralModel(nn.Module):
    def __init__(self, nfeatures, X_means, X_stds, y_mean, y_std, model_type='nonlinear'):
        super(NeuralModel, self).__init__()
        self.X_means = X_means
        self.X_stds = X_stds
        self.y_mean = y_mean
        self.y_std = y_std
        if model_type == 'nonlinear':
            self.fc_in = nn.Sequential(
                    nn.Linear(nfeatures, 200),
                    nn.ReLU(),
                    nn.Dropout(),
                    nn.Linear(200, 200),
                    nn.ReLU(),
                    nn.Dropout(),
                    nn.Linear(200, 1))
        elif model_type == 'linear':
            self.fc_in = nn.Sequential(nn.Linear(nfeatures,1))
        else:
            raise Exception('Unknown model type: {}'.format(model_type))
    
    def forward(self, x):
        return self.fc_in(x)

    def predict(self, X):
        self.eval()
        self.zero_grad()
        tX = autograd.Variable(torch.FloatTensor((X - self.X_means[np.newaxis,:]) / self.X_stds[np.newaxis,:]), requires_grad=False)
        tYhat = self.forward(tX)[:,0]
        return (tYhat.data.numpy() * self.y_std) + self.y_mean

def fit_nn(X, y, nepochs=100, batch_size=10, val_pct=0.1,
                 verbose=False, lr=3e-4, weight_decay=0.01,
                 model_type='nonlinear'):
    import uuid
    tmp_file = '/tmp/' + str(uuid.uuid4())

    # Standardize the features (helps with gradient propagation)
    Xstd = X.std(axis=0)
    Xstd[Xstd == 0] = 1 # Handle constant features
    tX = autograd.Variable(torch.FloatTensor((X - X.mean(axis=0,keepdims=True)) / Xstd[np.newaxis, :]), requires_grad=False)
    tY = autograd.Variable(torch.FloatTensor((y - y.mean()) / y.std()), requires_grad=False)

    # Create train/validate splits
    indices = np.arange(X.shape[0], dtype=int)
    np.random.shuffle(indices)
    train_cutoff = int(np.round(len(indices)*(1-val_pct)))
    train_indices = indices[:train_cutoff]
    validate_indices = indices[train_cutoff:]

    model = NeuralModel(X.shape[1], X.mean(axis=0), Xstd, y.mean(), y.std(), model_type=model_type)

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
            tYhat = model(tX[tidx])[:,0]

            # MSE loss
            loss = ((tY[tidx] - tYhat)**2).sum()

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

            # Run the model and get the prior predictions
            tYhat = model(tX[tidx])[:,0]

            # MSE loss
            loss = ((tY[tidx] - tYhat)**2).sum()

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


class NeuralCvModel:
    def __init__(self, models, folds):
        self.models = models
        self.folds = folds

    def predict(self, X):
        y = np.zeros(X.shape[0])
        for fold, model in zip(self.folds, self.models):
            y[fold] = model.predict(X[fold])
        return y

def fit_cv(X, y, nepochs=100, batch_size=10, val_pct=0.1,
                 verbose=False, lr=3e-4, weight_decay=0.01,
                 model_type='nonlinear', nfolds=5):
    folds = create_folds(X, nfolds)
    models = []
    for fold_idx,fold in enumerate(folds):
        print('Fitting model {}'.format(fold_idx))
        mask = np.ones(X.shape[0], dtype=bool)
        mask[fold] = False
        models.append(fit_nn(X[mask], y[mask], nepochs=nepochs, batch_size=batch_size, val_pct=val_pct,
                              verbose=verbose, lr=lr, weight_decay=weight_decay,
                              model_type=model_type))
    return NeuralCvModel(models, folds)

