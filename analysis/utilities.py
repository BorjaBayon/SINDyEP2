"""
Functions for data pre(post)-processing: measuring correlation, plotting data, printing models...
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.signal import butter, sosfiltfilt
from scipy.integrate import odeint
from sklearn.metrics import mean_squared_error


def print_model(coefs, feature_names, score = 0., inc_prob = 0., var_name = "var"):
    """
    Prints the symbolic ODE of the target variable given its sparse coefficient vector and feature library names.

    Parameters
    ----------
    coefs : ndarray of shape (n_features,)
        Model coefficients
    feature_names : list of shape (n_features,)
        List of strings containing the feature names corresponding to each term of coefs
    score : int, default = "var"
        Model score
    var_name : string, default = "var"
        Name of the target variable 
    """
    inds_nonzero = np.ravel(np.nonzero(coefs)) # to only print non-zero terms
    text = "[%.2f] [%.2f] "%(score, inc_prob) + var_name + "_dot = "
    for ind in inds_nonzero:
        text += "+ %.2e %s " % (coefs[ind], feature_names[ind])
    print(text)


def print_hierarchy_f(coef_list, n_terms, score, inc_prob, feature_names_list, var_name = "var"):
    """
    Prints model hierarchy for var_dot.

    Parameters
    ----------
    coefs : ndarray of shape (n_models, n_features)
        Model coefficients
    feature_names_list : list of shape (n_features,)
        List of strings containing the feature names corresponding to each term of coefs
    score : int, default = "var"
        Model score
    var_name : string, default = "var"
        Name of the target variable 
    """
    print("\n#############################################\nDisplaying identified models:\n")
    
    n_models = coef_list.shape[0]
    idx = sorted(range(n_models), key = lambda k: (n_terms[k], score[k])) # sort for printing in order
    print_coef = coef_list[idx]
    print_score = score[idx]
    print_inc_prob = np.array(inc_prob)[idx]

    for j in range(n_models):
        print_model(print_coef[j], feature_names_list, print_score[j], print_inc_prob[j], var_name)

    print("#############################################\n")




### Generate, add noise and plot synthetic data

def generate_data(time_span, n_points, type, state0 = 0,
                rho = 28.0, sigma = 10.0, beta = 8.0 / 3.0,
                xi = 1, w0 = 3,
                a = 2/3, b = 1, c = 1, d = 1/3, 
                mu = 5,
                A = 0.3, B = 4, F = 8, G = 1,
                Amp = 0, f = 5):
    """
    Generate n_points during time_span of certain data set for initial condition state0. 
    Types are "lorenz", "harm_osc", "lotka_volterra", "vanderpol", "hadley". External sinusoidal modulation can be added to the integrated trajectory.
    Model parameters and external modulation amplitude and frequency are modifiable.
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
    
    def f_vanderpol(state, t):
        x, y = state
        return mu*(x - x**3/3 - y), x/mu

    def f_hadley(state, t):
        x, y, z = state
        return -y**2 - z**2 - A*(x - F), x*y - B*x*z - y + G, B*x*y + x*z - z

    if type == "lorenz":
        if state0 is 0: state0 = [8.0, 7.0, 15.0]
        func = f_lorenz

    elif type == "harm_osc":
        if state0 is 0: state0 = [1.0, -1.0]
        func = f_harmosc

    elif type == "lotka_volterra":
        if state0 is 0: state0 = [1.0, 1.0]
        func = f_lv

    elif type == "vanderpol":
        if state0 is 0: state0 = [1.0, 1.0]
        func = f_vanderpol

    elif type == "hadley":
        if state0 is 0: state0 = [-10.0, 0.0, 37.0]
        func = f_hadley

    t = np.arange(0.0, time_span, time_span/n_points)
    X = odeint(func, state0, t).T*(1 + Amp*np.sin(f*t))

    return X.T, t

def add_noise(data, percentage):
    """
    Input: data array, percentage divided by 100
    """
    rmse = mean_squared_error(data, np.zeros(data.shape), squared=False)
    data_noisy = data + np.random.normal(loc=0, scale = rmse*percentage, size = data.shape)
    return data_noisy

def filter_data(data, cutoff, fs, order=5):
    """
    Butterworth-style low pass filter.
    """
    sos = butter(order, cutoff, fs=fs, btype='low', analog=False, output='sos')
    y = sosfiltfilt(sos, data) #passes filter in both directions to avoid introducing phase delays
    return y

def plot_data(data, fs, n_plotted=2000, bins=100, names=0, compare = False, data2 = None):
    """
    Given array of data with data.shape[1] variables sampled at rate fs, plots time series, derivative 
    Fourier spectrum and histogram (variable distribution) for each.
    
    Can use names to pass array with labels for each variable.
    """
    if data.ndim is 1: data = data.reshape(-1, 1)
    time_steps = data.shape[0]; n_var = data.shape[1]

    fig, axs = plt.subplots(n_var, 5, figsize=(24, 3*n_var), squeeze=False)
    axs[0,0].set_title("First "+str(n_plotted)+" points of variable"); axs[0,1].set_title("First "+str(n_plotted)+" points of derivative")
    axs[0,2].set_title("Fourier Spectrum"); axs[0,3].set_title("Distribution"); axs[0,4].set_title("Phase plot")
    for i in range(n_var):
        var = data[:,i]
        f, P = FourierTransform(var, fs, plot=False)
        if type(names) != int: axs[i, 0].plot(var[:n_plotted], label = names[i]); axs[i, 0].legend()
        else: axs[i, 0].plot(var[:n_plotted])
        axs[i, 1].plot(np.diff(var)[:n_plotted]*fs)
        axs[i, 2].semilogx(f, P)
        axs[i, 3].hist(var, bins=bins)
        axs[i, 4].scatter(var[:n_plotted], np.diff(var)[:n_plotted]*fs, s=3)

        if compare == True: 
            var2 = data2[:,i]
            f, P = FourierTransform(var2, fs, plot=False)
            axs[i, 0].plot(var2[:n_plotted], alpha = 0.5)
            axs[i, 1].plot(np.diff(var2)[:n_plotted]*fs, alpha = 0.5)
            axs[i, 2].semilogx(f, P, alpha=0.5)
            axs[i, 3].hist(var2, bins=bins, alpha=0.5)
            axs[i, 4].scatter(var2[:n_plotted], np.diff(var2)[:n_plotted]*fs, s=3, alpha=0.5)


def measure_correlation(data, column_names):
    pddata = pd.DataFrame(data)
    pddata.columns = column_names

    corr = pddata.corr(method = "pearson")
    fig, axs = plt.subplots(ncols=1, figsize=(20, 15))
    sns.heatmap(np.abs(corr), annot = True, ax=axs, vmin=0)


def FourierTransform(Xt, Fs, plot=True):
    """
    Input: 1D Time series Xt, Sampling frequency Fs,
    Output: frequency range, real part of frequency power spectrum and Fourier Spectrum plot if plot=True
    """
    L = len(Xt)
    f = Fs*np.arange(int(L/2))/L
    Xf = np.fft.fft(Xt)

    P2 = abs(Xf/L)
    P1 = P2[0:int(L/2)]
    P1[1:-2] = 2*P1[1:-2]
    P1[0] = 0
    P1 = P1/np.max(P1)
    if plot == True: plt.semilogx(f.T, P1)
    return f, P1