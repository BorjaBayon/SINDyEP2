"""
Functions
"""
import numpy as np
from sklearn.preprocessing import normalize
from sklearn.linear_model import lasso_path


def ALASSO_path(Theta, X_dot, n_alphas = 100, eps = 0.001):
    """
    Obtains the Adaptive LASSO/Reweighted L1-norm solution path.
    Normalizes Theta and X_dot, obtains weights as the inverse of the OLS estimate,
    and computes the LASSO path introducing the weights through Theta.
    Output coef_path are not renormalized.

    IN: Theta [n_points, n_features], X_dot [n_points]
    OUT: alpha_list [n_alphas], coef_path[n_features, n_alphas]
    """
    X_dotn, normXd = normalize(X_dot.reshape(-1,1), return_norm=True, axis=0)
    Thetan, normTheta = normalize(Theta, return_norm=True, axis=0)
    
    x0 = np.linalg.lstsq(Thetan, X_dotn, rcond=None)[0]
    W = 1 / np.abs(x0.flatten())
    Thetan_s = Thetan/W
    alpha_list, coef_path, _ = lasso_path(Thetan_s, X_dotn, params={"max_iter": 13000},
                                            n_alphas=n_alphas, eps = eps)

    return alpha_list, coef_path[0]




## Identify, remove duplicates, fit and find optimal supports

def identify_unique_supports(coefs, n_max_features = 10):
    """
    Given a coef list of form [active_coefs, lambda_reg], returns all combinations of individual supports
    Returns list of supports of len lambda_reg 
    """
    supports = []
    for row in coefs.T:
        support_lambdai = list(np.nonzero(row)[0]) # identify index of non-zero elements

        if len(support_lambdai) < n_max_features and len(support_lambdai) != 0: 
            supports.append(support_lambdai) # store if support is in the range of interest
    
    supports = remove_duplicates(supports) # remove duplicate indexes
    return supports
    
def remove_duplicates(lst):
    unique_lst = []

    for elem in lst:
        # if the element is not already in the unique list, add it
        if elem not in unique_lst:
            unique_lst.append(elem)
    
    # return the new list of unique elements
    return unique_lst

def fit_supports(Theta, X_dot, supports):
    """
    Use Ordinary Least Squares to obtain the unbiased model estimates for each support

    IN: Theta [n_points, n_features], X_dot [n_points], supports [n_supports, n_features]
    OUT: coefs[n_supports, n_features], score [n_supports], n_terms [n_supports]
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
        plt.scatter(n_terms, 1 - score, color="black", s=12)
    
    return coefs, score, n_terms

def find_optimal_support(coefs, score, n_terms):
    """
    Computes Pareto distance for each model and returns the one that minimizes said distance
    For print_hierarchy: 0 > don't print; 1 > print Pareto front; 2 > print all ranked by terms and score
    
    IN: coefs[n_supports, n_features], score [n_supports], n_terms [n_supports]
    OUT: optimal_coef[n_features], index_min [1]
    """
    pareto_distance = ( (1 - score)**2 + (n_terms / (np.max(n_terms + 1)))**2 )**0.5
    var_p_d_min = pareto_distance.min()
    index_min = int(np.argwhere(pareto_distance == var_p_d_min)[0])

    optimal_coef = coefs[index_min, :]
    return optimal_coef, index_min