"""
Functions related to computing weak forms of libraries and data
"""
import numpy as np

def compute_weak_forms(Theta, X, t):
    """
    Uses the trapezoidal rule and difference to compute for every t X_sm and Theta_sm:

    X_t - X_0 = \beta \int_{0}^{t}\Theta(X_\tau )d\tau
    X_sm = \beta Theta_sm

    IN: Theta [n_points, n_features], X [n_points], t [n_points]
    OUT: Theta [n_points - 1, n_features], X [n_points - 1]
    """
    X_sm = X[1:,:] - X[0,:]

    Theta_sm = np.zeros((Theta.shape[0] - 1, Theta.shape[1]))
    for i in range(Theta.shape[1]):
        Theta_sm[:,i] = np.cumsum(Theta[0:-1, i] + Theta[1:, i], axis=0)*t[1]/2

    return Theta_sm, X_sm

def windowed_generation(Theta, X, t, n_windows = 1000, min_w_size = 100):
    """
    Uses the trapezoidal rule and difference to compute X_sm and Theta_sm:

    X_t1 - X_t0 = \beta \int_{t0}^{t1}\Theta(X_\tau )d\tau
    X_sm = \beta Theta_sm

    for n_windows random t0 and t1, with a minimum window size of min_w_size.

    IN: Theta [n_points, n_features], X [n_points, n_var], t [n_points]
    OUT: Theta_sm [n_windows, n_features], X_sm [n_windows, n_var]
    """
    n_points = X.shape[0]
    Theta_sm = np.zeros((n_windows, Theta.shape[1]))
    X_sm = np.zeros((n_windows, X.shape[1]))

    for window in range(n_windows):
        w_size = min_w_size + int(np.random.random()*n_points/10)
        idx0 = int(np.random.random()*(n_points - w_size))
        idx1 = idx0 + w_size

        Theta_sm[window] = np.trapz(Theta[idx0:idx1], t[idx0:idx1], axis=0)
        X_sm[window] = X[idx1] - X[idx0]

    return Theta_sm, X_sm