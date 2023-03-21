"""
Functions related to the main workflow of the sparse identification process
"""
from CORE import ALASSO_path
from CORE import identify_unique_supports
from CORE import fit_supports
from CORE import find_optimal_support
from CORE import remove_duplicate_supports
from analysis import *
from preprocessing import *

import numpy as np
import warnings

def run_search(Theta, X_dot, var, 
               n_bootstraps = 100, n_features_to_drop = 2, n_max_features = 5,
               feature_names_list = [" "], print_hierarchy = 0):
    """
    Performs a search of the ALASSO solution (regularization) path, indentifies supports and fits them with OLS, returning the optimal model
    Can bootstrap data (n_bootstraps) and features (n_features_to_drop) randomly in the feature selection (support finding) process. 
    Only keeps supports with a number of features below n_max_features 
    IN: Theta [n_points, n_features], X_dot [n_points], supports [n_supports, n_features]

    Parameters
    ----------
    Theta : ndarray of shape (n_samples, n_features)
        Library of features, typically polynomial
    X_dot : ndarray of shape (n_samples, n_targets)
        Derivatives of target variables
    var : int
        Position of target variable in X_dot, i.e. X_target = X_dot[:, var]
    n_features_to_drop : int, default = 2
        Number of columns to delete from Theta in feature bootstrapping
    n_max_features : int, default = 5
        Maximum of number of features when looking for supports
    feature_names_list : list of shape (n_features,), default = [" "]
        List of strings containing the feature names corresponding to each term of coefs
    print_hierarchy : int, default = 0
        Flag to pass to print every model found (2), only the optimal ones (1) or to not print anything (0)

    Returns
    -------
    opt_coefs : ndarray of shape (n_features,)
        Unbiased coefficients of the Pareto optimal model
    opt_score : float
        R2 score of the optimal model
    """
    coefs = np.zeros((Theta.shape[1], 100, n_bootstraps))
    supports = []

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")


        for i in range(n_bootstraps):
            if n_bootstraps != 1:
                Theta_new, X_dot_new, inds = bootstrapping(Theta, X_dot, n_features_to_drop = n_features_to_drop)
            else:
                Theta_new, X_dot_new, inds = Theta, X_dot, np.arange(Theta.shape[1]) # if n_bootstraps = 1, the single bootstrap contains the entire dataset


            alphas, coefs[inds, :, i] = ALASSO_path(Theta_new, X_dot_new[:,var].reshape(-1,1),)
            supports += identify_unique_supports(coefs[:,:,i], n_max_features = n_max_features)
        
    supports = remove_duplicate_supports(supports)
    coef_list, score, n_terms = fit_supports(Theta, X_dot[:,var], supports)
    opt_coefs, index_min = find_optimal_support(coef_list, score, n_terms)
    
    print_hierarchy_f(print_hierarchy, coef_list, n_terms, score, feature_names_list)
    
    return opt_coefs, score[index_min]