"""
Functions for data pre(post)-processing: measuring correlation, plotting data, printing models...
"""
import numpy as np
from scipy.integrate import odeint
from sklearn.metrics import mean_squared_error

def print_model(coefs, score, features_names_list, var_name):
    inds_nonzero = np.ravel(np.nonzero(coefs))
    text = "[%.2f] "%(score) + var_name + "_dot = "
    for i in range(len(inds_nonzero)):
        text += "+ %8.2f %s " % (coefs[inds_nonzero[i]], features_names_list[inds_nonzero[i]])
    print(text)

def print_hierarchy_f(print_hierarchy, coef_list, n_terms, score, feature_names_list, var_name = "var"):
    """
    Depending on print_hierarchy keyword:
    If print_hierarchy = 0, doesn't print anything
    If print_hierarchy = 1, prints the Pareto front models (best model for every n_terms)
    If print_hierarchy = 2, prints every model found

    IN: coef_list [n_models, n_features], n_terms [n_models], score [n_models], 
    feature_names_list [n_features]
    OUT: 
    """
    if print_hierarchy == 0:
        return 
    
    elif print_hierarchy == 1:

        costs = np.c_[n_terms, 1 - score]
        is_efficient = np.ones(costs.shape[0], dtype = bool)
        for i, c in enumerate(costs):
            if is_efficient[i]:
                is_efficient[is_efficient] = np.any(costs[is_efficient]<c, axis=1)  # Keep any point with a lower cost
                is_efficient[i] = True  # And keep self

        coef_list = coef_list[is_efficient]; n_terms = n_terms[is_efficient]; score = score[is_efficient]
        ### CHECK: should not, but does this modify original coef_list?
    
    print("\n#############################################\nDisplaying identified models:\n")
    
    n_models = coef_list.shape[0]
    idx = sorted(range(n_models), key = lambda k: (n_terms[k], score[k])) # sort for printing in order
    print_coef = coef_list[idx]
    print_score = score[idx]

    for j in range(n_models):
        print_model(print_coef[j], print_score[j], feature_names_list, var_name)

    print("#############################################\n")




### Generate and add noise to synthetic data

def generate_data(time_span, n_points, type, xi=1, w0=3,
                  a = 2/3, b = 1, c = 1, d = 1/3, A = 0, f = 5, state0 = 0):
    """
    Generate n_points during time_span of certain data set for initial condition state0. 
    Types are "lorenz", "harm_osc", "hall_thruster", "lotka_volterra".
    "hall_thuster" gives n_points of ion and neutral density of a point in the middle of a 300V HET discharge.
    Oscillator parameters (damping xi, frequency w0) are modifyable.
    Lotka-Volterra parameters (a, b, c, d) and modulation (A, f) are also modifyable.
    """
    def f_lorenz(state, t):
        x, y, z = state  # Unpack the state vector
        return sigma*(y - x), x*(rho - z) - y, x*y - beta*z  # Derivatives

    def f_harmosc(state, t):
        x, v = state
        return v, -2*xi*w0*v - w0**2*x

    def f_lv(state, t):
        x, y = state
        return a*x - b*x*y, c*x*y - d*y

    if type == "lorenz":
        rho = 28.0; sigma = 10.0; beta = 8.0 / 3.0
        if state0 is 0: state0 = [8.0, 7.0, 15.0]
        func = f_lorenz

    elif type == "harm_osc":
        if state0 is 0: state0 = [1.0, -1.0]
        func = f_harmosc

    elif type == "lotka_volterra":
        if state0 is 0: state0 = [1.0, 1.0]
        func = f_lv

    t = np.arange(0.0, time_span, time_span/n_points)
    X = odeint(func, state0, t).T*(1 + A*np.sin(f*t))

    return X.T, t

def add_noise(data, percentage):
    """
    Input: data array, percentage divided by 100
    """
    rmse = mean_squared_error(data, np.zeros(data.shape), squared=False)
    data_noisy = data + np.random.normal(loc=0, scale = rmse*percentage, size = data.shape)
    return data_noisy