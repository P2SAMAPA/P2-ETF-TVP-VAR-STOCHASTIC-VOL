import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

class RecursiveTVPVAR:
    """
    Time‑varying parameter VAR estimated via recursive OLS with exponential forgetting.
    """
    def __init__(self, n_vars, n_lags, forgetting_factor=0.96):
        self.n_vars = n_vars
        self.n_lags = n_lags
        self.ff = forgetting_factor
        self.reset()

    def reset(self):
        n = self.n_vars * self.n_lags + 1  # +1 for intercept
        # P = (X'X)^{-1} initialised as large diagonal (uninformative)
        self.P = np.eye(n) * 1e6
        self.beta = np.zeros((self.n_vars, n))  # each row is coefficients for one equation

    def update(self, y_t, X_t):
        """
        y_t: (n_vars,) vector of current observations.
        X_t: (n_vars, n) design matrix? Actually we need separate for each equation.
        We'll loop over equations.
        """
        for i in range(self.n_vars):
            x = X_t[i, :].reshape(-1, 1)  # column vector
            # Kalman gain: K = P x / (x' P x + sigma^2)
            # But we treat sigma^2 as 1? Alternatively, we use recursive least squares with forgetting.
            # Simpler: use RLS with forgetting factor:
            # beta_new = beta_old + K * (y_t[i] - x' beta_old)
            # K = P_old * x / (ff + x' P_old x)
            y = y_t[i]
            x = x.flatten()
            # Prediction error
            pred = np.dot(self.beta[i], x)
            err = y - pred
            # Gain
            denom = self.ff + np.dot(x, self.P @ x)
            K = (self.P @ x) / denom
            # Update beta
            self.beta[i] = self.beta[i] + err * K
            # Update P
            self.P = (self.P - np.outer(K, np.dot(x, self.P))) / self.ff

def fit_tvp_var(data, lags=2, forgetting=0.96):
    """
    data: DataFrame with columns: ETF returns + macro variables (endogenous).
    Returns time‑varying coefficient matrix for the last period and the covariance matrix of residuals.
    """
    n_vars = data.shape[1]
    n = len(data)
    # Build design matrix: [1, y_{t-1}, ..., y_{t-p}]
    Y = data.values
    # For recursive estimation, we simulate online
    tvp = RecursiveTVPVAR(n_vars, lags, forgetting)
    tvp.reset()
    # Need to precompute lagged values: we'll do a moving window
    # For each t from lags+1 to n, use the most recent lags as features
    # Simpler: use all data up to t to estimate current beta via recursive least squares.
    for t in range(lags+1, n):
        # Construct X_t for each equation (same design for all equations)
        # Features: intercept + lags of all variables
        x = np.ones(1 + n_vars * lags)
        idx = 1
        for lag in range(1, lags+1):
            x[idx:idx+n_vars] = Y[t-lag, :]
            idx += n_vars
        # For each equation, the same x is used
        X_mat = np.tile(x, (n_vars, 1))
        y_t = Y[t, :]
        tvp.update(y_t, X_mat)
    # The final beta matrix (n_vars x (1 + n_vars*lags)) contains time‑varying coefficients.
    beta_last = tvp.beta
    # Compute residual covariance from last few steps
    residuals = np.zeros((n - lags - 1, n_vars))
    for t in range(lags+1, n):
        x = np.ones(1 + n_vars * lags)
        idx = 1
        for lag in range(1, lags+1):
            x[idx:idx+n_vars] = Y[t-lag, :]
            idx += n_vars
        pred = beta_last @ x
        residuals[t - lags - 1, :] = Y[t, :] - pred
    resid_cov = np.cov(residuals, rowvar=False)
    return beta_last, resid_cov

def impulse_response(beta, resid_cov, shock_var_index, horizon=5, shock_scale=1.0):
    """
    Compute impulse response of all variables to a one‑standard‑deviation shock in `shock_var_index`.
    beta: coefficient matrix (n_vars x (1 + n_vars*lags))
    resid_cov: covariance matrix of residuals.
    """
    n_vars = beta.shape[0]
    lags = (beta.shape[1] - 1) // n_vars
    # Cholesky decomposition for orthogonalised shock
    chol = np.linalg.cholesky(resid_cov)
    # Shock vector: only the specified variable gets a shock
    shock = np.zeros(n_vars)
    shock[shock_var_index] = shock_scale * chol[shock_var_index, shock_var_index]
    # Build companion form
    n_states = n_vars * lags
    F = np.zeros((n_states, n_states))
    # First block: coefficient matrices
    for i in range(lags):
        beta_block = beta[:, 1 + i*n_vars : 1 + (i+1)*n_vars]
        F[:n_vars, i*n_vars:(i+1)*n_vars] = beta_block.T
    # Shift block
    for i in range(lags-1):
        F[n_vars*(i+1):n_vars*(i+2), n_vars*i:n_vars*(i+1)] = np.eye(n_vars)
    # Impulse response vector
    irf = np.zeros((horizon+1, n_vars))
    # Initial impact
    irf[0, :] = shock
    # State vector
    state = np.zeros(n_states)
    state[:n_vars] = shock
    for h in range(1, horizon+1):
        state = F @ state
        irf[h, :] = state[:n_vars]
    return irf

def tvp_var_score(returns_df, macro_df, shock_var='VIX', lags=2, forgetting=0.96, horizon=5):
    """
    Combine ETF returns and macro data, estimate TVP‑VAR, and return impulse response
    of each ETF to a one‑standard‑deviation shock in the specified macro variable.
    """
    # Align dates
    common_idx = returns_df.index.intersection(macro_df.index)
    if len(common_idx) < 100:
        return {ticker: 0.0 for ticker in returns_df.columns}
    data = pd.concat([returns_df.loc[common_idx], macro_df.loc[common_idx]], axis=1)
    # Ensure no NaNs
    data = data.dropna()
    if data.empty:
        return {ticker: 0.0 for ticker in returns_df.columns}
    n_vars = data.shape[1]
    # Find index of shock variable
    if shock_var not in data.columns:
        return {ticker: 0.0 for ticker in returns_df.columns}
    shock_idx = list(data.columns).index(shock_var)
    beta, resid_cov = fit_tvp_var(data, lags, forgetting)
    irf = impulse_response(beta, resid_cov, shock_idx, horizon, shock_scale=1.0)
    # For each ETF, the response at horizon (e.g., 5 days) is the score
    # We'll use the cumulative response over horizon? Use final horizon.
    scores = {}
    for ticker in returns_df.columns:
        if ticker in data.columns:
            col_idx = list(data.columns).index(ticker)
            scores[ticker] = float(irf[horizon, col_idx])
        else:
            scores[ticker] = 0.0
    return scores
