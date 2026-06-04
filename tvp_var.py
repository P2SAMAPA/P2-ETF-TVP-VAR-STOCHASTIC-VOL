import numpy as np

def tvp_var_score(returns, shock_var=None, macro_shock=None, lambda_=None, forgetting=0.96):
    """
    Compute the time‑varying coefficient of a macro shock on ETF returns.
    Accepts either `shock_var` or `macro_shock` as the shock series.
    `lambda_` is used as forgetting factor (1 - lambda_ is the forgetting rate).
    """
    # Handle lambda_ parameter
    if lambda_ is not None:
        forgetting = lambda_
    # Determine the shock series
    if shock_var is not None:
        macro_shock = shock_var
    if macro_shock is None or len(macro_shock) == 0:
        return 0.0
    n = len(returns)
    if n < 5 or len(macro_shock) < n:
        return 0.0
    # Align lengths
    returns = returns[:n]
    macro_shock = macro_shock[:n]
    # Create regressors: constant + lagged return (1 day) + macro shock (contemporaneous)
    X = np.column_stack([np.ones(n-1), returns[:-1], macro_shock[1:]])
    y = returns[1:]
    # Recursive least squares with forgetting
    beta = np.zeros(3)
    P = np.eye(3) * 1e6
    for t in range(len(y)):
        x = X[t].flatten()
        denom = forgetting + x @ (P @ x)
        if denom <= 0:
            continue
        K = (P @ x) / denom
        err = y[t] - x @ beta
        beta = beta + err * K
        P = (P - np.outer(K, x @ P)) / forgetting
    # The coefficient for macro shock is beta[2]
    irf = beta[2] if len(beta) > 2 else 0.0
    return float(np.clip(irf, -1.0, 1.0))
