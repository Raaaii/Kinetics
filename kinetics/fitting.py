# Kinetics/fitting.py

from lmfit import minimize, Parameters, report_fit


def run_fit(objective_func, initial_params, data_list, model_name="Model"):
    """
    General wrapper for running lmfit on any model.

    Parameters:
    - objective_func: callable for computing residuals
    - initial_params: lmfit.Parameters object with starting values
    - data_list: list of input data tuples
    - model_name: optional name to display during fit

    Returns:
    - result: lmfit.MinimizerResult
    """
    print(f"\n=== Fitting {model_name} ===")
    result = minimize(objective_func, initial_params, args=(data_list,))
    print(f"\n=== Fit Report for {model_name} ===")
    print(report_fit(result))
    return result
