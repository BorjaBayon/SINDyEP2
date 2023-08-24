"""
Functions related to the main workflow of the sparse identification process
"""
from CORE import ALASSOp
from CORE import LASSOp	
from CORE import STLSQp
from CORE import STRidgep
from CORE import identify_unique_supports
from CORE import fit_supports
from CORE import find_Pareto_front
from CORE import find_Pareto_knee
from CORE import plot_Pareto_front
from CORE import remove_duplicate_supports
from analysis import *
from preprocessing import *
from sklearn.preprocessing import normalize
from sklearn.model_selection import train_test_split
from numpy.linalg import norm

import numpy as np
import warnings


def run_search(Theta, X_dot, var, optimizer = ALASSOp, scorer="R2",
               n_bootstraps = 1, n_features_to_drop = 0, n_max_features = 5,
               eps = 1e-5, n_alphas = 100, n_thresholds = 10,
               normalize_data = True,
               feature_names_list = [" "], print_hierarchy = 0):
    """
    Performs a search of the optimizer solution path, indentifies supports and fits them with OLS, returning the optimal model.
    Can bootstrap data (n_bootstraps) and features (n_features_to_drop) randomly in the feature selection process. 
    Only keeps supports with a number of features below n_max_features.
  
    Parameters
    ----------
    Theta : ndarray of shape (n_samples, n_features)
        Library of features, typically polynomial
    X_dot : ndarray of shape (n_samples, n_targets)
        Derivatives of target variables
    var : int
        Position of target variable in X_dot, i.e. X_target = X_dot[:, var]
    optimizer : function, default = ALASSOp
        Function to use for the optimization process. Must be one of ALASSOp, LASSOp, STLSQp or STRidgep
    scorer : string, default = "R2"
        Scorer to use for the Pareto seach process. Must be one of "R2", "CV"
    n_bootstraps : int, default = 1
        Number of bootstraps to generate. If n_bootstraps = 1, the entire dataset is used for the search.
    n_features_to_drop : int, default = 2
        Number of columns to delete from Theta in feature bootstrapping. If n_features_to_drop = 0, no culling is performed.
    n_max_features : int, default = 5
        Maximum of number of features when looking for supports
    eps : float, default = 1e-5. 
        Length of the path; eps = alpha_min / alpha_max where alpha_max = np.sqrt( np.sum(X_dot) / (n_samples) ).max()
    n_alphas: int, default=100
        Number of alphas along the regularization path
    n_thresholds : int, default = 10
        Number of thresholds to use in STLSQp and STRidgep
    normalize_data : bool, default = True
        Flag to normalize both sides of the minimization before fitting
    feature_names_list : list of shape (n_features,), default = [" "]
        List of strings containing the feature names corresponding to each term of coefs
    print_hierarchy : int, default = 0
        Flag to pass to print every model found (2), only the optimal ones (1) or to not print anything (0)

    Returns
    -------
    coef_list : ndarray of shape (n_models, n_features)
        List of coefficients of the models found
    score : ndarray of shape (n_models,)
        Score of the models found
    n_terms : ndarray of shape (n_models,)
        Number of terms of the models found
    front_idx : ndarray of shape (n_models,)
        Indices of the models in the Pareto front
    knee_idx : ndarray of shape (n_models,)
        Indices of the models in the Pareto knee
    """
    if optimizer == STLSQp or optimizer == STRidgep:
        n_thresholds = n_thresholds
    else:
        n_thresholds = 1
    coefs = np.zeros((Theta.shape[1], n_thresholds*n_alphas, n_bootstraps))
    idx = np.arange(Theta.shape[0])
    (idx_train, idx_test) = train_test_split(idx, test_size=0.25)
    supports = []

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")

        for i in range(n_bootstraps):
            if n_bootstraps != 1:
                Theta_new, X_dot_new, inds = bootstrapping(Theta[idx_train], X_dot[idx_train], n_features_to_drop = n_features_to_drop, )
            else:
                Theta_new, X_dot_new, inds = Theta[idx_train], X_dot[idx_train], np.arange(Theta.shape[1]) # if n_bootstraps = 1, the single bootstrap contains the entire dataset

            if normalize_data is True:
                X_dot_new, normXd = normalize(X_dot_new, return_norm=True, axis=0)
                Theta_new, normTheta = normalize(Theta_new, return_norm=True, axis=0)

            alphas, coefs[inds, :, i] = optimizer(Theta_new, X_dot_new[:,var].reshape(-1,1),
                                                    eps = eps, n_alphas = n_alphas, n_thresholds = n_thresholds,)

            if normalize_data is True:
                coefs[inds,:,i] = coefs[inds,:,i] * (normXd[var]/normTheta)[:,np.newaxis] # renormalize
            supports += identify_unique_supports(coefs[:,:,i], n_max_features = n_max_features)
        
    supports, inc_prob = remove_duplicate_supports(supports)
    # perform model fitting and scoring on the original dataset, not the bootstrapped one
    coef_list, score, n_terms = fit_supports(Theta, X_dot[:,var], idx_train, idx_test, supports, score_type=scorer)
    min_n_terms = np.min(n_terms)
    if min_n_terms > 1:
        print("Warning: no models with 1 term found. Minimum number of terms is {}".format(min_n_terms))

    if len(coef_list) == 0: # if no supports are found, return all-zeros model. Alternative: include all-zeros models from beggining
        coef_list = np.zeros((1, Theta.shape[1]))
        score = np.zeros(1)
        n_terms = np.zeros(1)
        inc_prob = np.zeros(1)

    front_coefs, front_idx = find_Pareto_front(coef_list, score, n_terms, n_max_terms = n_max_features)
    knee_coefs, knee_idx, pareto_distances = find_Pareto_knee(coef_list, score, n_terms)

    if print_hierarchy == 1:
       # print_hierarchy_f(coef_list[front_idx], (n_terms)[front_idx], score[front_idx], np.array(inc_prob)[front_idx], feature_names_list)
        print_hierarchy_f(coef_list[front_idx], (n_terms)[front_idx], score[front_idx], pareto_distances[front_idx], feature_names_list)
        plot_Pareto_front(coef_list, score, n_terms, inc_prob, front_idx, knee_idx)
    elif print_hierarchy == 2:
        print_hierarchy_f(coef_list, n_terms, score, inc_prob, feature_names_list)

    return coef_list, score, n_terms, front_idx, knee_idx