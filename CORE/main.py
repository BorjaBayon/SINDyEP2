"""
Functions related to the main workflow of the sparse identification process
"""
from CORE import ALASSO_path
from CORE import identify_unique_supports
from CORE import fit_supports
from CORE import find_Pareto_front
from CORE import find_Pareto_knee
from CORE import remove_duplicate_supports
from analysis import *
from preprocessing import *

import numpy as np
import warnings

def run_model_search(Theta, fTheta, X_dot, X, t, u = None,
               n_bootstraps = 100, n_features_to_drop = 2, n_max_features = 5,
               eps = 1e-3, n_alphas = 100,
               feature_names_list = [" "], print_hierarchy = 0):
    """
    Performs a search of the ALASSO solution (regularization) path for each target variable,
    identifies supports and fits them with OLS, returning the optimal model.

    Note that Theta and fTheta are passed as different arrays to still allow using the weak formulation
    just by inputting the weak Theta and X; fTheta is only used for model integration.
    
    Parameters
    ----------
    Theta : ndarray of shape (n_samples, n_features)
        Library of features, typically polynomial
    fTheta : callable(X, u)
        Computes Theta(X, u) such that given X[t] and u[t] it returns an array of shape (n_features, ).
    X_dot : ndarray of shape (n_samples, n_targets)
        Derivatives of target variables
    X : ndarray of shape (n_samples, n_var)
        Target and dependable variables.
    t : ndarray of shape (n_samples,)
        Time array.
    u : ndarray of shape (n_samples, n_control_var), default = None
        Control variables, if any.
    var : int
        Position of target variable in X_dot, i.e. X_target = X_dot[:, var]
    n_bootstraps : int, default = 100
        Number of bootrstraps to generate
    n_features_to_drop : int, default = 2
        Number of columns to delete from Theta in feature bootstrapping
    n_max_features : int, default = 5
        Maximum of number of features when looking for supports
    eps : float, default = 1e-5. 
        Length of the path; eps = alpha_min / alpha_max where alpha_max = np.sqrt( np.sum(X_dot) / (n_samples) ).max()
    n_alphas: int, default=100
        Number of alphas along the regularization path
    feature_names_list : list of shape (n_features,), default = [" "]
        List of strings containing the feature names corresponding to each term of coefs
    print_hierarchy : int, default = 0
        Flag to pass to print every model found (2), only the optimal ones (1) or to not print anything (0)

    Returns
    -------
    best_model : ndarray of shape (n_features,)
        Unbiased coefficients of the model with best score during integration
    best_score : float
        R2 score of the model with best score during integration
    opt_model : ndarray of shape (n_features,)
        Unbiased coefficients of the Pareto optimal model
    opt_score : float
        R2 score of the optimal model
    """

    n_targets = X_dot.shape[1]
    coef_array = []

    ## temporary
    opt_model = []
    opt_score = np.zeros(n_targets)

    for target in range(n_targets):
        opt_coefs, opt_score[target], front_coefs = run_search(Theta, X_dot, target,
               n_bootstraps, n_features_to_drop, n_max_features,
               eps, n_alphas,
               feature_names_list, print_hierarchy)
        
        coef_array.append(front_coefs)
        opt_model.append(opt_coefs)

    models_array = get_array_of_models(coef_array)
    score_int = model_integration(models_array, fTheta, X, t, u, n_windows=100)
    ordered_idx = sorted(np.arange(len(models_array)), key = lambda k: 1 - score_int[k])
    best_idx = ordered_idx[0] #model with the highest score
    best_model = models_array[best_idx]
    
    return best_model, score_int[best_idx], opt_model, opt_score


def run_search(Theta, X_dot, var,
               n_bootstraps = 100, n_features_to_drop = 2, n_max_features = 5,
               eps = 1e-5, n_alphas = 100,
               feature_names_list = [" "], print_hierarchy = 0):
    """
    Performs a search of the ALASSO solution (regularization) path, identifies supports and fits them with OLS, returning the optimal model

    Parameters
    ----------
    Theta : ndarray of shape (n_samples, n_features)
        Library of features, typically polynomial
    X_dot : ndarray of shape (n_samples, n_targets)
        Derivatives of target variables
    var : int
        Position of target variable in X_dot, i.e. X_target = X_dot[:, var]
    n_bootstraps : int, default = 100
        Number of bootrstraps to generate
    n_features_to_drop : int, default = 2
        Number of columns to delete from Theta in feature bootstrapping
    n_max_features : int, default = 5
        Maximum of number of features when looking for supports
    eps : float, default = 1e-5. 
        Length of the path; eps = alpha_min / alpha_max where alpha_max = np.sqrt( np.sum(X_dot) / (n_samples) ).max()
    n_alphas: int, default=100
        Number of alphas along the regularization path
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
    coefs = np.zeros((Theta.shape[1], n_alphas, n_bootstraps))
    supports = []

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")


        for i in range(n_bootstraps):
            if n_bootstraps != 1:
                Theta_new, X_dot_new, inds = bootstrapping(Theta, X_dot, n_features_to_drop = n_features_to_drop)
            else:
                Theta_new, X_dot_new, inds = Theta, X_dot, np.arange(Theta.shape[1]) # if n_bootstraps = 1, the single bootstrap contains the entire dataset


            alphas, coefs[inds, :, i] = ALASSO_path(Theta_new, X_dot_new[:,var].reshape(-1,1),
                                                    eps = eps, n_alphas = n_alphas)
            supports += identify_unique_supports(coefs[:,:,i], n_max_features = n_max_features)
        
    supports = remove_duplicate_supports(supports)
    coef_list, score, n_terms = fit_supports(Theta, X_dot[:,var], supports)
    front_coefs, front_idx = find_Pareto_front(coef_list, score, n_terms)
    opt_coefs, opt_idx = find_Pareto_knee(coef_list, score, n_terms)
    
    if print_hierarchy == 1:
        print_hierarchy_f(front_coefs, n_terms[front_idx], score[front_idx], feature_names_list)
    elif print_hierarchy == 2:
        print_hierarchy_f(coef_list, n_terms, score, feature_names_list)

    return opt_coefs, score[opt_idx], front_coefs