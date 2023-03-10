"""
Functions
"""

def ALASSO_path(Theta, X_dot, n_alphas = 100, eps = 0.001):
    X_dotn, normXd = normalize(X_dot.reshape(-1,1), return_norm=True, axis=0)
    Thetan, normTheta = normalize(Theta, return_norm=True, axis=0)
    
    x0 = np.linalg.lstsq(Thetan, X_dotn, rcond=None)[0]
    W = 1 / np.abs(x0.flatten())
    Thetan_s = Thetan/W
    alpha_list, coef_path, _ = lasso_path(Thetan_s, X_dotn, params={"max_iter": 13000},
                                            n_alphas=n_alphas, eps = eps)

    return alpha_list, coef_path[0]