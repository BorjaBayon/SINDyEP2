"""
Functions related to model ensembling and data bootstrapping
"""
import numpy as np

def bootstrapping(Theta, X_dot, n_features_to_drop = 2):
    """
    Given predictor X_dot and feature library Theta, returns bootstrap of both and indexes of kept features.
    The resample contains between 20% and 80% of the original sample with replacement.
    If n_features_to_drop = 0 feature bootstrapping is skipped and returned indexes correspond to all features.

    IN: Theta [n_points, n_features], X_dot [n_points, n_var]
    OUT: Theta_new [random, (n_features - n_features_to_drop)], X_dot[random], 
    keep_inds [(n_features - n_features_to_drop)]
    """
    n_samples = X_dot.shape[0]
    n_features = Theta.shape[1]
    
    ## Data bootstrapping
    n_points_per_boostrap = int((np.random.random()*0.6 + 0.2)*n_samples)
    rand_inds = np.random.choice(range(n_samples), n_points_per_boostrap, replace=True)
    X_dot_new = np.take(X_dot, rand_inds, axis=0)
    
    if n_features_to_drop > 0:
        ## Feature bootstrapping
        keep_inds = np.sort(np.random.choice(range(n_features), n_features - n_features_to_drop, replace=False,))
        Theta_new = np.take(Theta, rand_inds, axis=0)[:, keep_inds]
    else:
        keep_inds = np.arange(Theta.shape[1])
        Theta_new = np.take(Theta, rand_inds, axis=0)

    return Theta_new, X_dot_new, keep_inds