import numpy as np

def tvp_var_score(returns, shock_var=None, lambda_=0.96, **kwargs):
    """
    Compute the time‑varying coefficient of a macro shock on ETF returns.
    shock_var can be passed as second positional argument or keyword.
    """
    # If shock_var was passed as keyword, it might be in kwargs
    if shock_var is None and 'shock_var' in kwargs:
        shock_var = kwargs['shock_var']
    if shock_var is None:
        return 0.0
    n = len(returns)
    if n < 5 or len(shock_var) < n:
        return 0.0
    # Create regressors: constant + lagged return (1 day) + macro shock (contemporaneous)
    X = np.column_stack([np.ones(n-1), returns[:-1], shock_var[1:]])
    y = returns[1:]
    beta = np.zeros(3)
    P = np.eye(3) * 1e6
    forgetting = lambda_
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
    irf = np.clip(irf, -1.0, 1.0)
    return float(irf)
