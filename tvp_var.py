import numpy as np

def tvp_var_score(returns, shock_var, lambda_=0.96):
    """
    Compute the time‑varying coefficient of a macro shock on ETF returns.
    Uses recursive least squares with exponential forgetting.
    Returns the last coefficient (impulse response of return to macro shock).
    """
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
        if denom <= 1e-8:
            continue
        K = (P @ x) / denom
        err = y[t] - x @ beta
        beta = beta + err * K
        P = (P - np.outer(K, x @ P)) / forgetting
    # The coefficient for macro shock is beta[2]
    irf = beta[2] if len(beta) > 2 else 0.0
    return float(np.clip(irf, -1.0, 1.0))
