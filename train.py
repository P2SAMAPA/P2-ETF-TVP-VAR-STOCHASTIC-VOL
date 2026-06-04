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
    macro_window = macro_df.loc[ret_window.index] if macro_df is not None else None
    if macro_window is None:
        return None
    # Use first macro as shock variable (VIX)
    shock_var = config.MACRO_VARS[0] if config.MACRO_VARS else 'VIX'
    try:
        raw_scores = tvp_var_score(
            ret_window,
            macro_window,
            shock_var=shock_var,
            lags=config.VAR_LAG,
            forgetting=config.FORGETTING_FACTOR,
            horizon=config.IMPULSE_HORIZON
        )
    except Exception as e:
        print(f"    Error: {e}")
        return None
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
        print("Error: No macro data found. Exiting.")
        return
    results = {
        "run_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "windows": config.WINDOWS,
        "var_lag": config.VAR_LAG,
        "forgetting_factor": config.FORGETTING_FACTOR,
        "impulse_horizon": config.IMPULSE_HORIZON,
        "shock_variable": config.MACRO_VARS[0] if config.MACRO_VARS else "VIX",
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
