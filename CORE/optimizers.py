"""
Functions related to the optimization and support-finding
"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import normalize
from sklearn.linear_model import lasso_path
from sklearn.linear_model import ridge_regression
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.model_selection import cross_val_score




def ALASSOp(Theta, X_dot, eps = 1e-3, n_alphas = 100, n_thresholds = 1):
    """
    Obtains the Adaptive LASSO/Reweighted L1-norm solution path.
    Normalizes Theta and X_dot, obtains weights as the inverse of the OLS estimate,
    and computes the LASSO path introducing the weights through Theta, then renormalize.
    
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
    # X_dotn, normXd = normalize(X_dot.reshape(-1,1), return_norm=True, axis=0)
    # Thetan, normTheta = normalize(Theta, return_norm=True, axis=0)
    
    x0 = np.linalg.lstsq(Theta, X_dot, rcond=None)[0]
    W = 1 / np.abs(x0.flatten())
    Theta_s = Theta/W
    alpha_list, coef_path, _ = lasso_path(Theta_s, X_dot, params={"max_iter": 13000},
                                            n_alphas=n_alphas, eps = eps)

    return alpha_list, coef_path

def LASSOp(Theta, X_dot, n_alphas = 100, eps = 0.001, n_thresholds = 1):
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
    # X_dotn, normXd = normalize(X_dot.reshape(-1,1), return_norm=True, axis=0)
    # Thetan, normTheta = normalize(Theta, return_norm=True, axis=0)
    
    alpha_list, coef_path, _ = lasso_path(Theta, X_dot, params={"max_iter": 13000},
                                            n_alphas=n_alphas, eps = eps)

    return alpha_list, coef_path[0]

def STLSQ(Theta, X_dot, threshold):
    """
    Performs Sequentially Thresholded Least-Squares minimization.

    """
    
    big_ind = np.arange(Theta.shape[1])
    have_non_zero_coefs = True; coefs_change = True
    while coefs_change and have_non_zero_coefs:
        coef_i = np.zeros(Theta.shape[1])
        coef_i[big_ind] = np.linalg.lstsq(Theta[:,big_ind], X_dot, rcond=None)[0].flatten()
        coef_j = coef_i # see to another array for later comparison
        coef_j[np.abs(coef_j) < threshold] = 0 #threshold values below threshold
        big_ind = np.abs(coef_j) != 0

        if np.array_equal(coef_i, coef_j):
            coefs_change = False
        if np.count_nonzero(coef_i) == 0:
            have_non_zero_coefs = False
    
    # check if big_ind is all False
    if big_ind.any():
        coef_j[big_ind] = np.linalg.lstsq(Theta[:,big_ind], X_dot, rcond=None)[0].flatten() #unbias

    return coef_i


def STLSQp(Theta, X_dot, n_alphas = 100, eps = 0.001, n_thresholds = 100):
    coef_0 = np.linalg.lstsq(Theta, X_dot, rcond=None)[0].flatten()
    min_thrs = np.abs(coef_0).min()
    max_thrs = min_thrs/eps
    threshold_list = np.logspace(np.log10(min_thrs), np.log10(max_thrs), n_thresholds) # thresholds along the path
    
    coef_path = np.zeros((Theta.shape[1], n_thresholds*n_alphas))
    for j, threshold in enumerate(threshold_list):
        coef_path[:,j] = STLSQ(Theta, X_dot, threshold=threshold)

    return threshold_list, coef_path






def STRidge(Theta, X_dot, threshold, alpha = 0.05,
          max_iter = 100):
    """
    Performs Sequentially Thresholded Ridge Regression.

    """
    big_ind = np.arange(Theta.shape[1])
    have_non_zero_coefs = True; coefs_change = True
    while coefs_change and have_non_zero_coefs:
        coef_i = np.zeros(Theta.shape[1])
        coef_i[big_ind] = ridge_regression(Theta[:,big_ind], X_dot, alpha,)
        coef_j = coef_i # see to another array for later comparison
        coef_j[np.abs(coef_j) < threshold] = 0 #threshold values below threshold
        big_ind = np.abs(coef_j) != 0

        if np.array_equal(coef_i, coef_j):
            coefs_change = False
        if np.count_nonzero(coef_i) == 0:
            have_non_zero_coefs = False
    
    # check if big_ind is all False
    if big_ind.any():
        coef_j[big_ind] = np.linalg.lstsq(Theta[:,big_ind], X_dot, rcond=None)[0][0] #unbias

    return coef_i


def STRidgep(Theta, X_dot, n_alphas = 100, eps = 0.001, n_thresholds = 10):
    
    alpha_max = (np.sqrt( np.sum( (np.dot(Theta.T, X_dot))**2, axis=1) / (X_dot.shape[0]) ).max())*1e1
    # increase alpha_max until all coefficients are zero
    while np.count_nonzero(STRidge(Theta, X_dot, threshold=1e-3, alpha=alpha_max)) != 0:
        alpha_max *= 5.
        # if alpha_max is not big enough to make first model be all-zeroes, increase it

    alpha_min = eps*alpha_max
    alpha_list = np.logspace(np.log10(alpha_min), np.log10(alpha_max), n_alphas) # alphas along the path
    threshold_list = np.linspace(1e-3, 1, n_thresholds) # thresholds along the path

    coef_path = np.zeros((Theta.shape[1], n_alphas*n_thresholds))
    for j, threshold in enumerate(threshold_list):
        for i, alpha in enumerate(alpha_list):
            coef_path[:,j*n_alphas + i] = STRidge(Theta, X_dot, threshold=threshold, alpha=alpha)

    return alpha_list, coef_path
















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
    
    supports, inc_prob = remove_duplicate_supports(supports) # remove duplicate indexes
    return supports
    

def remove_duplicate_supports(lst):
    unique_elements = []
    counts = []
    unique_elements_dict = {}
    for element in lst:
        if isinstance(element, list):
            element = tuple(element)
        if element in unique_elements_dict:
            unique_elements_dict[element] += 1
        else:
            unique_elements_dict[element] = 1

    for key, value in unique_elements_dict.items():
        unique_elements.append(key)
        counts.append(value)

    return unique_elements, counts


def fit_supports(Theta, X_dot, supports, score_type="R2"):
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

    test_split = Theta.shape[0]//2

    for i, sup in enumerate(supports):
        coefs[i, sup] = np.linalg.lstsq(Thetan[:test_split,sup], X_dotn[:test_split].flatten(), rcond=None)[0]
        coefs[i] *= normXd[0]/(normTheta) # renormalize

        n_terms[i] = np.count_nonzero(coefs[i])

        if score_type == "R2":
            score[i] = r2_score(Theta[test_split:].dot(coefs[i]), X_dot[test_split:])
        elif score_type == "CV":
            score[i] = cross_val_score(LinearRegression(), Theta[:,sup], X_dot, cv=10).mean()
    
    return coefs, score, n_terms








#############################################################
## Pareto front analysis


def find_Pareto_front(coefs, score, n_terms, n_depth = 1, n_min_terms = 0, n_max_terms = 5):
    """
    For every model with n_terms returns the first n_depth with the biggest score

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
    return optimal_coef, index_min, pareto_distance


def plot_Pareto_front(coef_list, score, n_terms, inc_prob, front_idx, knee_idx):
    """
    Plots the Pareto front and the Pareto knee model.

    Parameters
    ----------
    coef_list : list of ndarrays of shape (n_features,)
        List of unbiased coefficients of the models in the Pareto front.
    score : ndarray of shape (n_supports,)
        R2 Score on the derivatives for each fitted support.
    n_terms : ndarray of shape (n_supports,)
        Size of each support, i.e. number of non-zero terms of each model.
    inc_prob : ndarray of shape (n_features,)
        Probability of inclusion of each feature in the model.
    front_idx : ndarray of shape (n_front_supports,)
        Indexes of Pareto front models in original coefs array.
    knee_idx : int
        Index of optimal support within coefs, to retrieve model score and n_terms.
    """

    score = np.maximum(0, score) # set negative scores to 0 for plotting purposes

    #fig, ax = plt.subplots(figsize=(20,7), nrows=1, ncols=2)
    fig, ax = plt.subplots(1,1, figsize=(9,7))
    ax.scatter(n_terms, 1 - score, c='white', marker='o', label='Other Models')
    ax.plot(np.append(0,n_terms[front_idx]), np.append(1, 1 - score[front_idx]), c='r', marker='o', label='Pareto front', linestyle='--', linewidth=3)
    ax.scatter(0, 1, c='r', marker='o') # add origin of Pareto
    ax.scatter(n_terms[knee_idx], 1 - score[knee_idx], c='r', marker='*', s = 250, label='Pareto knee')
    ax.grid()
    ax.set_xlabel('Number of terms')
    ax.set_ylabel('Error (1 - R2)')
    ax.set_ylim([0, 1.05])
    ax.legend()

    # ax[0].scatter(n_terms, 1 - score, c='k', marker='o', label='Models')
    # ax[0].scatter(n_terms[front_idx], 1 - score[front_idx], c='r', marker='o', label='Pareto front')
    # ax[0].scatter(0, 1, c='r', marker='o') # add origin of Pareto
    # ax[0].scatter(n_terms[knee_idx], 1 - score[knee_idx], c='r', marker='*', s = 200, label='Pareto knee')
    # ax[0].grid()
    # ax[0].set_xlabel('Number of terms')
    # ax[0].set_ylabel('Error (1 - R2)')
    # ax[0].legend()

    # ax[1].scatter(n_terms, inc_prob, c='k', marker='o', label='Models')
    # ax[1].scatter(n_terms[front_idx], np.array(inc_prob)[front_idx], c='r', marker='o', label='Pareto front')
    # ax[1].scatter(n_terms[knee_idx], np.array(inc_prob)[knee_idx], c='r', marker='*', s = 200, label='Pareto knee')
    # ax[1].grid()
    # ax[1].set_xlabel('Number of terms')
    # ax[1].set_ylabel('Probability of inclusion')
    # ax[1].legend()

