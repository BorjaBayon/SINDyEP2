"""
Functions for model search analysis: finding the Pareto front, building dynamical systems and 
testing on trajectories
"""
import numpy as np
import itertools
from sklearn.metrics import r2_score

def euler_integration(X0, beta, t_span, fTheta, u):
    """
    Implements i.e. X[t+1] = X[t] + X'[t]*dt with X'[t] = Theta(X, u)*beta
    """
    def fRHS(X, u, beta):
        return fTheta(X, u).dot(beta.T)
    
    X = np.zeros((t_span.shape[0], X0.shape[0]))
    X[0] = X0

    for t, dt in enumerate((t_span[1:] - t_span[:-1])):
        RHS = fRHS(X[t], u[t], beta).flatten()
        X[t+1] = np.maximum(np.minimum(X[t] + RHS*dt, 1e50), -1e50) #maxmin avoids overflow

    return X


def model_integration(coefs, fTheta, X, t, u = None, n_windows = 32, min_w_size = 100, max_w_size = 400):
    """
    Use random windows of integration to evaluate each model within coefs based on the data trajectories.

    Parameters
    ----------
    coefs : ndarray of shape (n_models, n_targets, n_features)
        Coefficients of the candidate models.
    fTheta : callable(X, u)
        Computes Theta(X, u) such that given X[t] and u[t] it returns an array of shape (n_features, ).
    X : ndarray of shape (n_samples, n_var)
        Target and dependable variables.
    t : ndarray of shape (n_samples,)
        Time array.
    u : ndarray of shape (n_samples, n_control_var), default = None
        Control variables, if any.
    n_windows : int, default = 32
        Number of integration windows.
    min_w_size : int, default = 100
        Minimum number of points taken for integration.
    max_w_size : int, default = 400
        Maximum number of points taken for integration.

    Returns
    -------
    R2int : ndarray of shape (n_models,)
        R2 Score on the trajectory for each fitted support
    """
    if u is None: u = np.ones_like(t)[:,None]
    
    n_models = coefs.shape[0]
    n_points = t.shape[0] 
    R2int = np.zeros(n_models)

    for window in range(n_windows):
        w_size = min_w_size + int(np.random.random() * (max_w_size - min_w_size))
        idx0 = int(np.random.random() * (n_points - w_size)) # avoids having a idx1 larger than n_points
        idx1 = idx0 + w_size
        t_span = t[idx0:idx1]

        for model in range(n_models):
            Xint = euler_integration(X[idx0], coefs[model], t_span, fTheta, u = u[idx0:idx1])
            R2 = np.maximum(0, r2_score(X[idx0:idx1] , Xint))
            R2int[model] += R2

    R2int = R2int / n_windows
    return R2int


def get_array_of_models(coef_array):
    """
    Obtains models for the full system dynamics by combining models for each of the target variables

    Parameters
    ----------
    coef_array : list of shape (n_targets, n_models(target), n_features)
        Coefficients of the candidate models for each target variable.

    Returns
    -------
    models_array : ndarray of shape (n_system_models, n_targets, n_features)
    """
    idx_list = [np.arange(len(coefs_list)) for coefs_list in coef_array] #list of candidate model index for each var_dot
    idx_comb_array = list(itertools.product(*idx_list)) #returns list of lists each with the index to take from each var_dot mode list
    models_array = np.array([[x[i] for i, x in zip(idx_comb_array[j], coef_array)] 
                      for j in range(len(idx_comb_array))]) 
    
    return models_array