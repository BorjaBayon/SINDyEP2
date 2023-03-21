"""
Functions related to model ensembling and data bootstrapping
"""
import numpy as np

def bootstrapping(Theta, X_dot, n_features_to_drop = 2):
    """
    Given predictor X_dot and feature library Theta, returns random sampling of both and indexes of kept features.
    The resample contains between 20% and 80% of the original sample, taken with replacement.
    If n_features_to_drop = 0 feature bootstrapping is skipped and returned indexes correspond to all features.

    Parameters
    ----------
    Theta : ndarray of shape (n_samples, n_features)
        Library of features, typically polynomial
    X_dot : ndarray of shape (n_samples, n_targets)
        Derivatives of target variables
    n_features_to_drop : int, default = 2
        Number of columns to delete from Theta in feature bootstrapping

    Returns
    -------
    Theta_new : ndarray of shape (n_samples_bootstrap, n_features - n_features_to_drop)
        Bootstrap selection taken from library of features
    X_dot_new : ndarray of shape (n_samples_bootstrap, n_targets)
        Bootstrap selection taken from derivatives of target variables
    keep_inds : ndarray of shape (n_features - n_features_to_drop,)
        Indexes of features not removed from library
    """
    n_samples = X_dot.shape[0]
    n_features = Theta.shape[1]
    
    ## Data bootstrapping
    n_samples_bootstrap = int((np.random.random()*0.6 + 0.2)*n_samples)
    rand_inds = np.random.choice(range(n_samples), n_samples_bootstrap, replace=True)
    X_dot_new = np.take(X_dot, rand_inds, axis=0)
    
    if n_features_to_drop > 0:
        ## Feature bootstrapping
        keep_inds = np.sort(np.random.choice(range(n_features), n_features - n_features_to_drop, replace=False,))
        Theta_new = np.take(Theta, rand_inds, axis=0)[:, keep_inds]
    else:
        keep_inds = np.arange(Theta.shape[1])
        Theta_new = np.take(Theta, rand_inds, axis=0)

    return Theta_new, X_dot_new, keep_inds