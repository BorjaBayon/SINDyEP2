# SINDyEP2
Sparse Identification of Nonlinear Dynamics algorithms as used by the EP2 team for system identification in Electric Propulsion systems


Structure of the code:
run_search() # //main.ipybn

  for n_bootstraps:
      bootstrapping() #can be toggled on and off, for creating random subsamples of data and features // bootstrapping.py
      ALASSO_path() #scans the regularization parameter to cover the whole range of ALASSO estimates // optimizers.py
      identify_unique_supports() #takes biased ALASSO estimates and returns the combinations of features appearing in them // optimizers.py

  remove_duplicates() #ensures that if one support appears in several bootstraps only one copy is stored // optimizers.py
  fit_supports() #fits the identified supports with OLS to obtain unbiased estimates // optimizers.py
  find_optimal_support() #computes Pareto distance, returns model that minimizes it // optimizers.py
  print_hierarchy_f() #prints all the models founds, just the best ones or prints nothing // utilities.py
 


Depending on the input of run_search can be run in weak or differential formulation. For the differential formulation:
X -> compute X_dot, Theta(X) -> run_search(Theta(X), X_dot) -> opt_coef, score of opt model
For the Weak formulation:
X -> compute Theta(X) -> compute weak forms X_sm, Theta(X)_sm -> run_search(Theta(X)_sm, X_sm) -> opt_coef, score of opt model

Weak forms can be computed with compute_weak_forms (single integration window) or windowed_generation (several randomly selected windows), both in weak_forms.py
