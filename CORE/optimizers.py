"""
Functions related to the optimization and support-finding
"""
import numpy as np
from sklearn.preprocessing import normalize
from sklearn.linear_model import lasso_path
from sklearn.metrics import r2_score

def ALASSO_path(Theta, X_dot, eps = 1e-3, n_alphas = 100):
    """
    Obtains the Adaptive LASSO/Reweighted L1-norm solution path.
    Normalizes Theta and X_dot, obtains weights as the inverse of the OLS estimate,
    and computes the LASSO path introducing the weights through Theta.
    NOTE: Output coef_path are not renormalized.
    
    Parameters
    ----------
    Theta : ndarray of shape (n_samples, n_features)
        Library of features, typically polynomial
    X_dot : ndarray of shape (n_samples,)
        Derivative of target variable
    eps : float, default=1e-3. 
        Length of the path; eps = alpha_min / alpha_max where alpha_max = np.sqrt( np.sum(X_dot) / (n_samples) ).max()
    n_alphas: int, default=100
        Number of alphas along the regularization path

    Returns
    -------
    alpha_list : ndarray of shape (n_alphas,)
        The alphas along the path where models are computed
    coef_path : ndarray of shape (n_features, n_alphas)
        Coefficients along the path
    """
    X_dotn, normXd = normalize(X_dot.reshape(-1,1), return_norm=True, axis=0)
    Thetan, normTheta = normalize(Theta, return_norm=True, axis=0)
    
    x0 = np.linalg.lstsq(Thetan, X_dotn, rcond=None)[0]
    W = 1 / np.abs(x0.flatten())
    Thetan_s = Thetan/W
    alpha_list, coef_path, _ = lasso_path(Thetan_s, X_dotn, params={"max_iter": 13000},
                                            n_alphas=n_alphas, eps = eps)

    return alpha_list, coef_path[0]

def LASSO_path(Theta, X_dot, n_alphas = 100, eps = 0.001):
    """
    Obtains the LASSO (Least Absolute Shrinkage and Selection Operator) solution path.
    Normalizes Theta and X_dot and computes the LASSO path.
    NOTE: Output coef_path are not renormalized.

    Parameters
    ----------
    Theta : ndarray of shape (n_samples, n_features)
        Library of features, typically polynomial
    X_dot : ndarray of shape (n_samples,)
        Derivative of target variable
    eps : float, default=1e-3. 
        Length of the path; eps = alpha_min / alpha_max where alpha_max = np.sqrt( np.sum(X_dot) / (n_samples) ).max()
    n_alphas: int, default=100
        Number of alphas along the regularization path

    Returns
    -------
    alpha_list : ndarray of shape (n_alphas,)
        The alphas along the path where models are computed
    coef_path : ndarray of shape (n_features, n_alphas)
        Coefficients along the path
    """
    X_dotn, normXd = normalize(X_dot.reshape(-1,1), return_norm=True, axis=0)
    Thetan, normTheta = normalize(Theta, return_norm=True, axis=0)
    
    alpha_list, coef_path, _ = lasso_path(Thetan, X_dotn, params={"max_iter": 13000},
                                            n_alphas=n_alphas, eps = eps)

    return alpha_list, coef_path[0]








#############################################################
## Identify, remove duplicates, fit and find optimal supports


def identify_unique_supports(coef_path, n_max_features = 10):
    """
    Given the distribution of coefficients along a solution path, returns every individual support found.
    
    Parameters
    ----------
    coef_path : ndarray of shape (n_features, n_alphas)
        Coefficients along the path
    n_max_features : int, default = 10
        Maximum of number of features when looking for supports

    Returns
    -------
    supports : ndarray of shape (n_supports,)
    """
    supports = []
    for row in coef_path.T:
        support_lambdai = list(np.nonzero(row)[0]) # identify index of non-zero elements

        if len(support_lambdai) < n_max_features and len(support_lambdai) != 0: 
            supports.append(support_lambdai) # store if support is in the range of interest
    
    supports = remove_duplicate_supports(supports) # remove duplicate indexes
    return supports
    

def remove_duplicate_supports(lst):
    unique_lst = []

    for elem in lst:
        # if the element is not already in the unique list, add it
        if elem not in unique_lst:
            unique_lst.append(elem)
    
    # return the new list of unique elements
    return unique_lst


def fit_supports(Theta, X_dot, supports):
    """
    Use Ordinary Least Squares to obtain the unbiased model estimates for each support.
    Normalizes the inputs for fitting but the returned coefficients are renormalized.

    Parameters
    ----------
    Theta : ndarray of shape (n_samples, n_features)
        Library of features, typically polynomial
    X_dot : ndarray of shape (n_samples,)
        Derivative of target variable
    supports : ndarray of shape (n_supports,)

    Returns
    -------
    coefs : ndarray of shape (n_supports, n_features)
        Unbiased coefficients along the path
    score : ndarray of shape (n_supports,)
        R2 Score on the derivatives for each fitted support
    n_terms : ndarray of shape (n_supports,)
        Size of each support, i.e. number of non-zero terms of each model
    """
    X_dotn, normXd = normalize(X_dot.reshape(-1,1), return_norm=True, axis=0)
    Thetan, normTheta = normalize(Theta, return_norm=True, axis=0)

    coefs = np.zeros((len(supports), Theta.shape[1]))
    score = np.zeros((len(supports)))
    n_terms = np.zeros((len(supports)))

    for i, sup in enumerate(supports):
        coefs[i, sup] = np.linalg.lstsq(Thetan[:,sup], X_dotn.flatten(), rcond=None)[0]
        coefs[i] *= normXd[0]/(normTheta) # renormalize

        score[i] = r2_score(Theta.dot(coefs[i]), X_dot)
        n_terms[i] = np.count_nonzero(coefs[i])
    
    return coefs, score, n_terms








#############################################################
## Pareto front analysis


def find_Pareto_front(coefs, score, n_terms, n_depth = 1, n_min_terms = 0, n_max_terms = 5):
    """
    For every model with n_terms returns only the one with the biggest score

    Parameters
    ----------
    coefs : ndarray of shape (n_supports, n_features)
        Unbiased coefficients along the path.
    score : ndarray of shape (n_supports,)
        R2 Score on the derivatives for each fitted support.
    n_terms : ndarray of shape (n_supports,)
        Size of each support, i.e. number of non-zero terms of each model.
    n_depth : int, default = 1
        Number of models to return per n_terms.
    n_min_terms : int, default = 0
        Minimum number of terms in returned models.
    n_max_terms : int, default = 5
        Maximum number of terms in returned models.

    Returns
    -------
    front_coef : ndarray of shape (n_front_supports, n_features)
        Unbiased coefficients of the models in the Pareto front.
    front_idx : ndarray of shape (n_front_supports,)
        Indexes of Pareto front models in original coefs array.
    """
    front_coefs = np.empty((0, coefs.shape[1]))
    front_idx = np.empty(0)

    for terms in np.unique(n_terms):
        if (terms >= n_min_terms) and (terms <= n_max_terms):            
            idx = np.argwhere(n_terms == terms).flatten()
            ordered_idx = sorted(idx, key = lambda k: 1 - score[k])

            depth = 0
            while depth < n_depth and depth < len(ordered_idx):
                front_coefs = np.append(front_coefs, coefs[ ordered_idx[depth] ][None,:], axis=0)
                front_idx = np.append(front_idx, int(ordered_idx[depth]))
                depth += 1
                 
    return front_coefs, front_idx.astype(int)


def find_Pareto_knee(coefs, score, n_terms):
    """
    Computes Pareto distance for each model and returns the one that minimizes said distance (the "Pareto knee" model):

    d_{Pareto, i} = \sqrt{ (1-R_i^2)^{2} +\left (\frac{n_{terms,i}}{n_{terms,max}+1}  \right )^2}

    Parameters
    ----------
    coefs : ndarray of shape (n_supports, n_features)
        Unbiased coefficients along the path
    score : ndarray of shape (n_supports,)
        R2 Score on the derivatives for each fitted support
    n_terms : ndarray of shape (n_supports,)
        Size of each support, i.e. number of non-zero terms of each model

    Returns
    -------
    optimal_coef : ndarray of shape (n_features,)
        Unbiased coefficients of the Pareto optimal model
    index_min : int
        Index of optimal support within coefs, to retrieve model score and n_terms
    """
    pareto_distance = ( (1 - score)**2 + (n_terms / (np.max(n_terms + 1)))**2 )**0.5
    var_p_d_min = pareto_distance.min()
    index_min = int(np.argwhere(pareto_distance == var_p_d_min)[0])

    optimal_coef = coefs[index_min, :]
    return optimal_coef, index_min