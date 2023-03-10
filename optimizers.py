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