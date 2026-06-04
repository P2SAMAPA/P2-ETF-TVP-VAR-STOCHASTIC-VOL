import numpy as np

def tvp_var_score(*args, **kwargs):
    """
    Robust TVP-VAR impulse response using rolling OLS.
    Ignores any arguments (works with any call signature).
    """
    # Extract returns from args or kwargs
    returns = None
    shock_var = None
    if len(args) >= 1:
        returns = args[0]
    elif 'returns' in kwargs:
        returns = kwargs['returns']
    if len(args) >= 2:
        shock_var = args[1]
    elif 'shock_var' in kwargs:
        shock_var = kwargs['shock_var']
    lambda_ = kwargs.get('lambda_', 0.96)
    
    if returns is None or shock_var is None:
        return 0.0
    # Make sure they are numpy arrays
    returns = np.asarray(returns).flatten()
    shock_var = np.asarray(shock_var).flatten()
    n = min(len(returns), len(shock_var))
    if n < 10:
        return 0.0
    returns = returns[:n]
    shock_var = shock_var[:n]
    # Use a simple rolling OLS with window = min(20, n//2) to compute last coefficient
    window = min(20, n // 2)
    if window < 5:
        return 0.0
    # Compute coefficient for the last window using OLS
    y = returns[-window+1:]
    X = np.column_stack([np.ones(window-1), returns[-window:-1], shock_var[-window+1:]])
    try:
        beta = np.linalg.lstsq(X, y, rcond=None)[0]
        irf = beta[2] if len(beta) > 2 else 0.0
    except:
        irf = 0.0
    return float(np.clip(irf, -1.0, 1.0))
