"""
Functions related to computing weak forms of libraries and data
"""
import numpy as np

def compute_weak_forms(Theta, X, t):
    """
    Uses the trapezoidal rule and difference to compute for every t X_sm and Theta_sm:

    X_t - X_0 = \beta \int_{0}^{t}\Theta(X_\tau )d\tau
    X_sm = \beta Theta_sm

    Parameters
    ----------
    Theta : ndarray of shape (n_samples, n_features)
        Library of features, typically polynomial
    X : ndarray of shape (n_samples,)
        Target variable
    t : ndarray of shape (n_samples,)
        Sampling time of the target variable trajectory

    Returns
    -------
    Theta_sm : ndarray of shape (n_samples - 1, n_features)
        Integrated library of features
    X_sm : ndarray of shape (n_samples - 1,)
        Target variable difference
    """
    X_sm = X[1:,:] - X[0,:]

    Theta_sm = np.zeros((Theta.shape[0] - 1, Theta.shape[1]))
    for i in range(Theta.shape[1]): # for every feature
        Theta_sm[:,i] = np.cumsum(Theta[0:-1, i], axis=0)*(t[1]-t[0])

    return Theta_sm, X_sm

def windowed_generation(Theta, X, t, n_windows = 1000, w_size = 100):
    """
    Uses the trapezoidal rule and difference to compute X_sm and Theta_sm:

    X_t1 - X_t0 = \beta \int_{t0}^{t1}\Theta(X_\tau )d\tau
    X_sm = \beta Theta_sm

    for n_windows random t0 and t1, with a minimum window size of min_w_size and maximum size of min_w_size + n_samples * 10%

    Parameters
    ----------
    Theta : ndarray of shape (n_samples, n_features)
        Library of features, typically polynomial
    X : ndarray of shape (n_samples, n_targets)
        Array of target variables
    t : ndarray of shape (n_samples,)
        Sampling time of the target variable trajectory
    n_windows : int, default = 1000
        Number of integration windows
    min_w_size : int, default = 100
        Number of points taken for an integration window

    Returns
    -------
    Theta_sm : ndarray of shape (n_windows, n_features)
        Concatenation of all the library of features integration
    X_sm : ndarray of shape (n_windows,)
        Concatenation of all the corresponding target variable differences
    """
    n_points = X.shape[0]
    Theta_sm = np.zeros((n_windows, Theta.shape[1]))
    X_sm = np.zeros((n_windows, X.shape[1]))

    for window in range(n_windows):
        idx0 = int(np.random.random()*(n_points - w_size)) # random window starting point
        idx1 = idx0 + w_size

        Theta_sm[window] = np.trapz(Theta[idx0:(idx1+1)], t[idx0:(idx1+1)], axis=0)
        X_sm[window] = X[idx1] - X[idx0]

    return Theta_sm, X_sm