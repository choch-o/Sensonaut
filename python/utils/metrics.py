import numpy as np

def mse_discrepancy(human_df, model_df, dep_vars):
    """Mean squared error between human and model across conditions."""
    group_cols = ["angle", "distractors", "num_cars"]
    
    # Human condition means
    human_means = human_df.groupby(group_cols)[dep_vars].mean()
    model_means = model_df.groupby(group_cols)[dep_vars].mean()
    
    # Align conditions
    aligned = human_means.join(model_means, lsuffix="_human", rsuffix="_model").dropna()
    
    # Compute MSE for each dependent variable
    mse = {}
    for var in dep_vars:
        diff = aligned[f"{var}_human"] - aligned[f"{var}_model"]
        mse[var] = np.mean(diff**2)
    
    # Optionally aggregate into a single score
    mse["overall"] = np.mean(list(mse.values()))
    return mse


from scipy.stats import wasserstein_distance, entropy

def dist_discrepancy(human_df, model_df, dep_var):
    """Wasserstein distance + KL divergence between human and model distributions."""
    human_vals = human_df[dep_var].values
    model_vals = model_df[dep_var].values
    
    # Wasserstein (Earth-Mover distance)
    wd = wasserstein_distance(human_vals, model_vals)
    
    # Histogram-based KL divergence
    hist_h, bins = np.histogram(human_vals, bins=30, density=True)
    hist_m, _ = np.histogram(model_vals, bins=bins, density=True)
    hist_h += 1e-12; hist_m += 1e-12
    kl = entropy(hist_h, hist_m)
    
    return {"wasserstein": wd, "kl": kl}



def accuracy_discrepancy(human_df, model_df):
    acc_h = human_df["estimate_correct"].mean()
    acc_m = model_df["estimate_correct"].mean()
    return abs(acc_h - acc_m)


def composite_discrepancy(human_df, model_df):
    dep_vars = ["duration", "distance", "accumulated_head_turns"]
    
    # Mean discrepancy
    mse = mse_discrepancy(human_df, model_df, dep_vars)
    
    # Accuracy discrepancy
    acc_diff = accuracy_discrepancy(human_df, model_df)
    
    # Weighted combination (tune weights as needed)
    return 0.4*mse["overall"] + 0.6*acc_diff


# --- Extended similarity metrics (level + pattern) and composite ---
from scipy.stats import pearsonr

DV_CONT = ["duration", "distance", "accumulated_head_turns"]
DV_ALL = DV_CONT + ["estimate_correct"]
IVS = ["angle", "distractors", "num_cars"]


def _align_conditions(human_df, model_df, dep_vars=DV_ALL, cond_cols=IVS):
    """Align condition means on common condition keys."""
    h_means = human_df.groupby(cond_cols)[dep_vars].mean().sort_index()
    m_means = model_df.groupby(cond_cols)[dep_vars].mean().sort_index()
    aligned = h_means.join(m_means, how="inner", lsuffix="_human", rsuffix="_model")
    return aligned


def _safe_hist_kl(x, y, bins=30):
    hist_p, bin_edges = np.histogram(x, bins=bins, density=True)
    hist_q, _ = np.histogram(y, bins=bin_edges, density=True)
    hist_p = hist_p + 1e-12
    hist_q = hist_q + 1e-12
    hist_p = hist_p / hist_p.sum()
    hist_q = hist_q / hist_q.sum()
    return float(entropy(hist_p, hist_q))


def _normalized(value, scale):
    if scale is None or np.isnan(scale) or scale <= 0:
        scale = 1.0
    return float(value) / float(scale)


def level_similarity_metrics(human_df, model_df):
    """Level metrics: normalized RMSE of condition means; WD and KL for distributions."""
    out = {}
    aligned = _align_conditions(human_df, model_df, dep_vars=DV_ALL)

    # RMSE between condition means, normalized by human std
    for dv in DV_ALL:
        diff = aligned[f"{dv}_human"] - aligned[f"{dv}_model"]
        rmse = np.sqrt(np.mean(diff**2)) if len(diff) else np.nan
        scale = np.nanstd(aligned[f"{dv}_human"]) if len(diff) else np.nan
        out[f"{dv}.rmse_means_norm"] = _normalized(rmse, scale)

    # Distribution-level distances for continuous vars
    for dv in DV_CONT:
        x = human_df[dv].dropna().values
        y = model_df[dv].dropna().values
        if len(x) and len(y):
            wd = wasserstein_distance(x, y)
            iqr = float(np.subtract(*np.percentile(x, [75, 25])))
            out[f"{dv}.wd_norm"] = _normalized(wd, iqr)
            out[f"{dv}.kl"] = _safe_hist_kl(x, y, bins=30)
        else:
            out[f"{dv}.wd_norm"] = np.nan
            out[f"{dv}.kl"] = np.nan
    return out


def _condition_mean_vector(df, dv):
    means = df.groupby(IVS)[dv].mean().sort_index()
    return means.values


def _contrast_effects(df, dv):
    g = df.groupby(IVS)[dv].mean()

    def get_mean(angle, distractors, num_cars):
        key = (angle, distractors, num_cars)
        return g.get(key, np.nan)

    effects = []
    # Distractors contrasts (2 vs 0, 4 vs 0)
    for (d_hi, d_lo) in [(2, 0), (4, 0)]:
        diffs = []
        for a in ["front", "side", "back"]:
            for n in [5, 7, 12]:
                hi = get_mean(a, d_hi, n)
                lo = get_mean(a, d_lo, n)
                if not (np.isnan(hi) or np.isnan(lo)):
                    diffs.append(hi - lo)
        effects.append(np.nanmean(diffs) if diffs else np.nan)

    # Num cars contrasts (7 vs 5, 12 vs 5)
    for (n_hi, n_lo) in [(7, 5), (12, 5)]:
        diffs = []
        for a in ["front", "side", "back"]:
            for d in [0, 2, 4]:
                hi = get_mean(a, d, n_hi)
                lo = get_mean(a, d, n_lo)
                if not (np.isnan(hi) or np.isnan(lo)):
                    diffs.append(hi - lo)
        effects.append(np.nanmean(diffs) if diffs else np.nan)

    # Angle contrasts (side vs front, back vs front)
    for (a_hi, a_lo) in [("side", "front"), ("back", "front")]:
        diffs = []
        for d in [0, 2, 4]:
            for n in [5, 7, 12]:
                hi = get_mean(a_hi, d, n)
                lo = get_mean(a_lo, d, n)
                if not (np.isnan(hi) or np.isnan(lo)):
                    diffs.append(hi - lo)
        effects.append(np.nanmean(diffs) if diffs else np.nan)

    return np.array(effects, dtype=float)


def pattern_similarity_metrics(human_df, model_df):
    """Pattern metrics: correlation of condition means; RMSE of contrast effects (normalized)."""
    out = {}
    for dv in DV_ALL:
        v_h = _condition_mean_vector(human_df, dv)
        v_m = _condition_mean_vector(model_df, dv)
        if len(v_h) == len(v_m) and len(v_h) > 2 and np.isfinite(v_h).all() and np.isfinite(v_m).all():
            r, _ = pearsonr(v_h, v_m)
        else:
            r = np.nan
        out[f"{dv}.corr_means"] = r

        e_h = _contrast_effects(human_df, dv)
        e_m = _contrast_effects(model_df, dv)
        if np.isfinite(e_h).all() and np.isfinite(e_m).all():
            rmse = np.sqrt(np.mean((e_h - e_m) ** 2))
            scale = np.nanstd(e_h)
            out[f"{dv}.effects_rmse_norm"] = _normalized(rmse, scale)
        else:
            out[f"{dv}.effects_rmse_norm"] = np.nan
    return out


def composite_score(metrics_dict, weights=None):
    """Composite score from level + pattern metrics (lower is better)."""
    default_weights = {
        "rmse_means_norm": 1.0,
        "wd_norm": 1.0,
        "kl": 0.25,
        "corr_means_to_dist": 2.0,
        "effects_rmse_norm": 2.0,
    }
    if weights is None:
        weights = default_weights

    pieces = []
    for key, val in metrics_dict.items():
        if val is None or (isinstance(val, float) and not np.isfinite(val)):
            continue
        if key.endswith(".corr_means"):
            # Convert correlation (higher better) to a distance in [0,1]
            dist = (1.0 - float(val)) / 2.0
            w = weights.get("corr_means_to_dist", 1.0)
            pieces.append(w * dist)
        elif key.endswith(".rmse_means_norm"):
            w = weights.get("rmse_means_norm", 1.0)
            pieces.append(w * float(val))
        elif key.endswith(".wd_norm"):
            w = weights.get("wd_norm", 1.0)
            pieces.append(w * float(val))
        elif key.endswith(".kl"):
            w = weights.get("kl", 1.0)
            pieces.append(w * float(val))
        elif key.endswith(".effects_rmse_norm"):
            w = weights.get("effects_rmse_norm", 1.0)
            pieces.append(w * float(val))
    if not pieces:
        return np.nan
    return float(np.nansum(pieces) / len(pieces))
