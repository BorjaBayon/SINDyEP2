"""
Functions for data pre(post)-processing: measuring correlation, plotting data, printing models...
"""
import numpy as np

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
        inds_nonzero = np.ravel(np.nonzero(print_coef[j]))
        text = "[%.2f] "%(print_score[j]) + var_name + "_dot = "
        for i in range(len(inds_nonzero)):
            text += "+ %8.2f %s " % (print_coef[j, inds_nonzero[i]], feature_names_list[inds_nonzero[i]])
        print(text)

    print("#############################################\n")