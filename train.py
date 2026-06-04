import os
import json
from datetime import datetime
import numpy as np
import pandas as pd
from huggingface_hub import HfApi
import config
import data_manager as dm
from tvp_var import tvp_var_score

def normalize_scores(score_dict):
    scores = np.array(list(score_dict.values()))
    scores = scores[np.isfinite(scores)]
    if len(scores) == 0:
        return {k: 0.0 for k in score_dict}
    min_s, max_s = scores.min(), scores.max()
    if max_s - min_s < 1e-12:
        return {k: 0.5 for k in score_dict}
    norm = (scores - min_s) / (max_s - min_s)
    tickers = list(score_dict.keys())
    return {tickers[i]: float(norm[i]) for i in range(len(norm))}

def run_for_window(returns, macro_df, window_days):
    if len(returns) < window_days:
        return None
    ret_window = returns.iloc[-window_days:]
    if macro_df is None or config.PRIMARY_MACRO not in macro_df.columns:
        return None
    # Align macro to same dates
    macro_window = macro_df.loc[ret_window.index]
    # Compute macro shock as first difference
    macro_series = macro_window[config.PRIMARY_MACRO].values
    if len(macro_series) < 2:
        return None
    shock_var = np.diff(macro_series)
    # Pad to same length as returns? We'll align with returns[1:] inside function.
    # The function expects shock_var length >= len(returns). But returns length = window_days.
    # Actually we need shock_var of length len(returns) (daily shocks). For simplicity, use diff.
    # But we'll repeat the last shock to match length? The function uses shock_var[1:], so it needs
    # len(shock_var) >= len(returns). Let's pad with zero at beginning.
    if len(shock_var) < len(ret_window):
        # Pad with zeros at start to match length
        pad = np.zeros(len(ret_window) - len(shock_var))
        shock_var = np.concatenate([pad, shock_var])
    raw_scores = {}
    for ticker in ret_window.columns:
        s = tvp_var_score(ret_window[ticker].values, shock_var, lambda_=config.LAMBDA)
        if not np.isfinite(s):
            s = 0.0
        raw_scores[ticker] = float(s)
    norm_scores = normalize_scores(raw_scores)
    sorted_norm = sorted(norm_scores.items(), key=lambda x: x[1], reverse=True)
    top_etfs = [{"ticker": t, "tvp_score_norm": s, "raw_score": raw_scores[t]} for t, s in sorted_norm[:config.TOP_N]]
    return {
        "window": window_days,
        "top_etfs": top_etfs,
        "all_scores_raw": raw_scores,
        "all_scores_norm": norm_scores
    }

def main():
    print("Loading master data...")
    dm.load_master_data()
    macro_df = dm.get_macro_data()
    if macro_df is None:
        print("Error: No macro data found.")
        return
    results = {
        "run_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "windows": config.WINDOWS,
        "primary_macro": config.PRIMARY_MACRO,
        "lambda": config.LAMBDA,
        "universes": {}
    }
    for uni_name in config.UNIVERSES.keys():
        print(f"Processing {uni_name}...")
        returns = dm.get_universe_returns(uni_name)
        if returns.empty:
            print("  No data -> skipping")
            continue
        all_window_results = []
        for w in config.WINDOWS:
            print(f"  Window {w} days")
            out = run_for_window(returns, macro_df, w)
            if out:
                all_window_results.append(out)
            else:
                print(f"    Failed for window {w}")
        best_data = all_window_results[-1] if all_window_results else None
        results["universes"][uni_name] = {
            "best_window_data": best_data,
            "all_windows": all_window_results
        }
    os.makedirs("output", exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_file = f"output/tvp_var_{timestamp}.json"
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved to {out_file}")
    api = HfApi(token=config.HF_TOKEN)
    try:
        api.upload_file(
            path_or_fileobj=out_file,
            path_in_repo=os.path.basename(out_file),
            repo_id=config.OUTPUT_REPO,
            repo_type="dataset"
        )
        print(f"Uploaded to {config.OUTPUT_REPO}")
    except Exception as e:
        print(f"Upload failed: {e}")

if __name__ == "__main__":
    main()
