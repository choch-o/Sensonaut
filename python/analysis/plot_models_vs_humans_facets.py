#!/usr/bin/env python3
"""
Pruned plotting script for comparing models vs humans.
Exports:
  1. Unified main effects plot (4x3: accuracy + duration/head_turns/distance)
  2. Correlation board (2x2: accuracy, search time, head turns, distance)
"""
import argparse
import glob
import os
import re
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from scipy import stats  # Added for correct T-statistics on small N

# Set Roboto as the default font, with fallbacks
def _set_font():
    """Try to use Roboto, fall back to other sans-serif fonts if not available."""
    available_fonts = set(f.name for f in fm.fontManager.ttflist)
    preferred_fonts = ['Roboto Condensed', 'DejaVu Sans', 'Arial', 'Helvetica', 'sans-serif']
    for font in preferred_fonts:
        if font in available_fonts or font == 'sans-serif':
            plt.rcParams['font.family'] = 'sans-serif'
            plt.rcParams['font.sans-serif'] = [font] + preferred_fonts
            break

_set_font()

ANGLE_ORDER = ["front", "side", "back"]
NUM_CARS_ORDER = [5, 7, 12]
DISTRACTORS_ORDER = [0, 2, 4]

# Font sizes
Y_LABEL_FONTSIZE = 18
TICK_LABEL_FONTSIZE = 16
LEGEND_FONTSIZE = 16
HUMAN_COLOR = "#e07a7a"
HUMAN_FILL_COLOR = "#f4c2c2"
SENSONAUT_COLOR = "#1b9e77"
SENSONAUT_FILL_COLOR = "#a1edd6"


# --- Participant column detection ---
PARTICIPANT_CANDIDATES = [
    "participant", "participant_id", "pid", "human_id", "subject", "worker_id"
]


def _find_participant_col(df: pd.DataFrame) -> str | None:
    for c in PARTICIPANT_CANDIDATES:
        if c in df.columns:
            return c
    return None


def _safe_suffix(name: str) -> str:
    base = os.path.basename(name)
    return re.sub(r"[^A-Za-z0-9._\\-]+", "_", base)


# --- Data loading ---
def load_human(human_csv: str) -> pd.DataFrame:
    df = pd.read_csv(human_csv)
    if "agent_type" in df.columns:
        df = df[df["agent_type"] == "human"].copy()
    return df


def load_models(mdf_glob: str) -> list[dict]:
    out = []
    for fp in sorted(glob.glob(mdf_glob)):
        try:
            df = pd.read_csv(fp)
            df = df[df["agent_type"] != "human"].copy()
        except Exception:
            continue
        name = _safe_suffix(os.path.splitext(os.path.basename(fp))[0])
        out.append({
            "name": name,
            "path": fp,
            "df": df,
        })
    return out


def _filter_duration_outliers_global(
    df: pd.DataFrame,
    z: float = 3.0,
    only_above: bool = True,
):
    """Drop duration outliers globally (not groupwise)."""
    if "duration" not in df.columns:
        return df.copy(), (0, 0.0)
    d = df["duration"].astype(float)
    mu = float(d.mean())
    sigma = float(d.std(ddof=0))
    if sigma <= 0 or not np.isfinite(sigma):
        return df.copy(), (0, 0.0)
    if only_above:
        keep_mask = d <= (mu + z * sigma)
    else:
        keep_mask = (d >= (mu - z * sigma)) & (d <= (mu + z * sigma))
    kept = df[keep_mask].reset_index(drop=True)
    n_removed = int((~keep_mask).sum())
    pct_removed = 100.0 * n_removed / max(1, len(df))
    return kept, (n_removed, pct_removed)


# --- Helper functions ---
def _nice_ylabel(dv: str) -> str:
    if dv == "duration":
        return "Search time (s)"
    if dv == "distance":
        return "Displacement (m)"
    if dv == "accumulated_head_turns":
        return "Head turns (deg)"
    if dv == "accuracy":
        return "Accuracy"
    return dv


def _human_band_max(human_df: pd.DataFrame, dv: str, unit_col: str = None) -> float:
    """Return the maximum of (human mean + std) across full grid."""
    # We must respect the unit_col aggregation here too for consistency
    if unit_col and unit_col in human_df.columns:
         # Aggregate by subject first to get valid subject means
        g_sub = human_df.groupby(["angle", "num_cars", "distractors", unit_col])[dv].mean().reset_index()
        # Then get mean/std across subjects
        g = g_sub.groupby(["angle", "num_cars", "distractors"])[dv].agg(["mean", "std"]).dropna()
    else:
        # Fallback to raw aggregation
        g = human_df.groupby(["angle", "num_cars", "distractors"])[dv].agg(["mean", "std"]).dropna()
    
    if g.empty:
        return float(human_df[dv].max() if dv in human_df else 1.0)
    
    upper = (g["mean"] + g["std"]).values
    if len(upper) == 0:
        return float(human_df[dv].max() if dv in human_df else 1.0)
    return float(np.nanmax(upper))


def _mean_ci_by_group(df: pd.DataFrame, dv: str, var: str, order, unit_col: str = None) -> pd.DataFrame:
    """
    Aggregate mean/count/std + SEM and 95% CI for dv grouped by var, reindexed to order.
    
    CRITICAL CHANGE: 
    If unit_col is provided, we first aggregate by [var, unit_col] to get the MEAN per subject.
    Then we compute statistics across subjects. This ensures N = Number of Subjects (12), not Trials (120).
    """
    if df.empty or (dv not in df.columns) or (var not in df.columns):
        return pd.DataFrame({
            "mean": [np.nan] * len(order),
            "count": [0] * len(order),
            "std": [0.0] * len(order),
            "se": [0.0] * len(order),
            "ci95_lo": [np.nan] * len(order),
            "ci95_hi": [np.nan] * len(order),
        }, index=order)

    # --- STEP 1: Aggregate by Subject (if applicable) ---
    if unit_col and unit_col in df.columns:
        # Collapse repeated measures: Take the mean for each subject in each condition
        # Result: One row per subject per condition
        grouped_data = df.groupby([var, unit_col])[dv].mean().reset_index()
        # Now we group by 'var' to get the stats across subjects
        g = grouped_data.groupby(var)[dv].agg(["mean", "count", "std"]).reindex(order)
    else:
        # Fallback: No subject column, treat every row as independent (Method 0)
        g = df.groupby(var)[dv].agg(["mean", "count", "std"]).reindex(order)

    g = g.fillna({"count": 0, "std": 0})

    # --- STEP 2: Calculate CI with correct N and T-value ---
    def _calc(row):
        c = row["count"] # This is now N_subjects (e.g., 12)
        s = row["std"]   # This is now SD_between_subjects
        
        if c > 1 and np.isfinite(s):
            se = s / math.sqrt(c)
            # Use T-distribution for small N (e.g., N=12 -> t ~ 2.20)
            # If N is large (>30), this converges to 1.96 naturally
            t_crit = stats.t.ppf(0.975, df=c-1) 
            delta = t_crit * se
        else:
            se = 0.0
            delta = 0.0
            
        return pd.Series({"se": se, "ci95_lo": row["mean"] - delta, "ci95_hi": row["mean"] + delta})

    out = g.join(g.apply(_calc, axis=1))
    return out


def _accuracy_samples(df: pd.DataFrame, unit_col: str | None, filters: dict) -> np.ndarray:
    """Return accuracy values aggregated per unit (participant/map) for the given filters."""
    if df.empty or "estimate_correct" not in df.columns:
        return np.array([])
    sub = df
    for k, v in filters.items():
        if k not in sub.columns:
            return np.array([])
        sub = sub[sub[k] == v]
    if sub.empty:
        return np.array([])
    
    # Aggregation for boxplots: We want 1 point per subject
    if unit_col and unit_col in sub.columns:
        acc = sub.groupby(unit_col)["estimate_correct"].mean()
        return acc.dropna().values.astype(float)
    return sub["estimate_correct"].dropna().astype(float).values


def _style_box(bp, edge_color, face_color):
    for box in bp['boxes']:
        box.set(color=edge_color, facecolor=face_color, alpha=0.9, linewidth=1.2)
    for whisker in bp['whiskers']:
        whisker.set(color=edge_color, linewidth=1.0)
    for cap in bp['caps']:
        cap.set(color=edge_color, linewidth=1.0)
    for median in bp['medians']:
        median.set(color=edge_color, linewidth=1.5)


def _unit_column(df: pd.DataFrame) -> str | None:
    cand = _find_participant_col(df)
    if cand:
        return cand
    if "map_id" in df.columns:
        return "map_id"
    return None


# --- Unified Main Effects Plot ---
def export_unified_main_effects(human_df: pd.DataFrame, models: list[dict], out_path: str):
    """
    Create a unified 4x3 figure showing main effects across conditions.
    ALL metrics (including Accuracy) are now plotted as Line Plots + CI Ribbons
    for visual consistency and trend comparison.
    """
    # Concatenate all agent models
    agent_list = [m["df"] for m in models if isinstance(m.get("df"), pd.DataFrame)]
    agent_df = pd.concat(agent_list, ignore_index=True) if agent_list else pd.DataFrame()

    human_unit = _unit_column(human_df)
    agent_unit = _unit_column(agent_df)

    # Define DVs. Map the display name "Accuracy" to the column "estimate_correct"
    # Format: (column_name, display_label, y_limit_fixed_tuple_or_None)
    metrics = [
        ("estimate_correct", "Accuracy", (0.0, 1.02)), # Fixed scale for accuracy (adjust 0.4 if needed)
        ("duration", "Search time (s)", None),
        ("accumulated_head_turns", "Head turns (deg)", None),
        ("distance", "Displacement (m)", None)
    ]

    specs = [
        ("angle", ANGLE_ORDER, "Angle"),
        ("num_cars", NUM_CARS_ORDER, "Number of Objects"),
        ("distractors", DISTRACTORS_ORDER, "Distractors"),
    ]

    # Create figure with 4 rows x 3 cols
    fig, axes = plt.subplots(4, 3, figsize=(8, 11))

    for row_idx, (dv_col, dv_label, ylim_fixed) in enumerate(metrics):
        
        # Calculate dynamic ylim if not fixed (for duration/dist/turns)
        if ylim_fixed is None:
            # We check if the column exists in human_df
            if dv_col in human_df.columns:
                band_hi = _human_band_max(human_df, dv_col, unit_col=human_unit) * 1.35
                ylim = (0, band_hi)
            else:
                ylim = (0, 1) # Fallback
        else:
            ylim = ylim_fixed

        for col_idx, (var, order, xlabel) in enumerate(specs):
            ax = axes[row_idx, col_idx]
            xs = range(len(order))

            # --- 1. HUMAN DATA (Ribbon + Line) ---
            if dv_col in human_df.columns:
                # Use the Method 1 (Subject-Aggregated) CI Calculator
                h = _mean_ci_by_group(human_df, dv_col, var, order, unit_col=human_unit)
                h_mean = h["mean"].values
                h_ci_lo = h["ci95_lo"].values
                h_ci_hi = h["ci95_hi"].values

                # Ribbon (Confidence Interval)
                ax.fill_between(xs, h_ci_lo, h_ci_hi,
                                color=HUMAN_FILL_COLOR, alpha=0.5, linewidth=0, zorder=1)
                # Mean Line
                ax.plot(xs, h_mean, color=HUMAN_COLOR, lw=2.5, ls=":",
                       marker="^", markeredgecolor='white', markeredgewidth=1.0, markersize=12, alpha=0.7, zorder=5)

            # --- 2. AGENT DATA (Ribbon + Line) ---
            if dv_col in agent_df.columns:
                m = _mean_ci_by_group(agent_df, dv_col, var, order, unit_col=agent_unit)
                m_mean = np.asarray(m["mean"].values, dtype=float)
                m_se = np.asarray(m["se"].values, dtype=float)
                m_ci_lo = np.asarray(m["ci95_lo"].values, dtype=float)
                m_ci_hi = np.asarray(m["ci95_hi"].values, dtype=float)

                # Ribbon (Confidence Interval)
                ax.fill_between(xs, m_ci_lo, m_ci_hi,
                                color=SENSONAUT_FILL_COLOR, alpha=0.8, linewidth=0, zorder=3.0)
                # Mean Line
                ax.plot(xs, m_mean, color=SENSONAUT_COLOR, lw=0.5,
                       marker="o", markeredgecolor='white', markeredgewidth=1.0, markersize=8, alpha=0.7, zorder=5)

            # --- Formatting ---
            ax.set_xticks(xs)
            ax.set_ylim(ylim)
            # Set 5 ticks on y-axis
            ax.yaxis.set_major_locator(plt.MaxNLocator(5))
            ax.grid(True, alpha=0.3)
            ax.tick_params(axis='both', labelsize=TICK_LABEL_FONTSIZE)

            # X-Axis Labels (Only on bottom row)
            if row_idx == 3:
                ax.set_xticklabels([str(x) for x in order])
                ax.set_xlabel(xlabel, fontsize=Y_LABEL_FONTSIZE, labelpad=10)
            else:
                ax.set_xticklabels([])

            # Y-Axis Labels (Only on leftmost column)
            if col_idx == 0:
                # ha='center' centers the text around its midpoint, va='center' for vertical
                ax.set_ylabel(dv_label, fontsize=Y_LABEL_FONTSIZE, ha='center', va='center')
                ax.yaxis.set_label_coords(-0.32, 0.5)  # 0.5 = vertically centered
            else:
                # Remove y-axis labels and tick labels for columns 2 and 3
                ax.set_ylabel("")
                ax.tick_params(axis='y', labelleft=False)

    # Create universal legend at center top
    # Custom legend handles showing line + shaded CI
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    # Order handles for 2x2 layout: row1=[Human mean, Agent mean], row2=[Human CI, Agent CI]
    legend_handles = [
        Line2D([0], [0], color=HUMAN_COLOR, lw=2.5, ls=":", marker="^",
               markeredgecolor='white', markeredgewidth=1.0, markersize=12, alpha=0.7, label="Human (mean)"),
        Patch(facecolor=HUMAN_FILL_COLOR, alpha=0.5, label="Human (95% CI)"),
        Line2D([0], [0], color=SENSONAUT_COLOR, lw=1.0, marker="o",
        markeredgecolor='white', markeredgewidth=1.0, markersize=8, alpha=0.8, label="Agent (mean)"),
        Patch(facecolor=SENSONAUT_FILL_COLOR, alpha=0.8, label="Agent (95% CI)"),
    ]

    fig.legend(handles=legend_handles,
               loc='upper center',
               bbox_to_anchor=(0.5, 1.01),
               ncol=4,
               frameon=True,
               fontsize=LEGEND_FONTSIZE,
               handlelength=1.7,
               handletextpad=0.4,
               columnspacing=1.0)

    plt.tight_layout(rect=[0, 0, 1, 0.96])  # Leave room at top for 1-row legend
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"[unified main effects] Saved {out_path}")


# --- Correlation Board (2x2) ---

def export_correlation_board(human_df: pd.DataFrame, models: list[dict], out_path: str, agg: str = "median"):
    """
    Create a 2x2 figure showing per-map correlation between human and agent for:
    - Accuracy (proportion correct per map)
    - Search time (duration)
    - Head turns (accumulated_head_turns)
    - Distance
    Each panel shows scatter with identity line and fit line with R^2.
    Legend is placed outside each plot.
    """

    # Concatenate all agent models
    agent_list = [m["df"] for m in models if isinstance(m.get("df"), pd.DataFrame)]
    agent_df = pd.concat(agent_list, ignore_index=True) if agent_list else pd.DataFrame()
    # DVs to plot (accuracy is computed separately)
    dvs = [
        ("accuracy", "Accuracy"),
        ("duration", "Search time (s)"),
        ("accumulated_head_turns", "Head turns (deg)"),
        ("distance", "Displacement (m)"),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(9, 9))
    axes = axes.flatten() # Flatten to iterate easily
    for ax_idx, (dv, dv_label) in enumerate(dvs):
        ax = axes[ax_idx]

        if dv == "accuracy":
            # Accuracy: compute mean (proportion correct) per map
            agg_label = "mean"
            if "estimate_correct" not in human_df.columns or "map_id" not in human_df.columns:
                ax.text(0.5, 0.5, "No accuracy data", ha="center", va="center", transform=ax.transAxes)
                ax.set_xlabel(f"Human {dv_label}")
                ax.set_ylabel(f"Agent {dv_label}")
                continue
            h_agg = human_df.groupby("map_id")["estimate_correct"].mean().rename("human")

            if "estimate_correct" not in agent_df.columns or "map_id" not in agent_df.columns:
                ax.text(0.5, 0.5, "No agent accuracy data", ha="center", va="center", transform=ax.transAxes)
                ax.set_xlabel(f"Human {dv_label}")
                ax.set_ylabel(f"Agent {dv_label}")
                continue
            a_agg = agent_df.groupby("map_id")["estimate_correct"].mean().rename("agent")
        else:
            # Continuous DVs: aggregate by map
            agg_label = agg

            if dv not in human_df.columns or "map_id" not in human_df.columns:
                ax.text(0.5, 0.5, f"No {dv} data", ha="center", va="center", transform=ax.transAxes)
                ax.set_xlabel(f"Human {dv_label}")
                ax.set_ylabel(f"Agent {dv_label}")
                continue

            g_h = human_df.groupby("map_id")[dv]
            h_agg = (g_h.mean() if agg == "mean" else g_h.median()).rename("human")
            if dv not in agent_df.columns or "map_id" not in agent_df.columns:
                ax.text(0.5, 0.5, f"No agent {dv} data", ha="center", va="center", transform=ax.transAxes)
                ax.set_xlabel(f"Human {dv_label}")
                ax.set_ylabel(f"Agent {dv_label}")
                continue
            g_a = agent_df.groupby("map_id")[dv]
            a_agg = (g_a.mean() if agg == "mean" else g_a.median()).rename("agent")

        # Join on map_id
        pairs = h_agg.to_frame().join(a_agg.to_frame(), how="inner").dropna()
        if pairs.empty:
            ax.text(0.5, 0.5, "No matched maps", ha="center", va="center", transform=ax.transAxes)
            ax.set_xlabel(f"Human {dv_label}")
            ax.set_ylabel(f"Agent {dv_label}")
            continue
        x = pairs["human"].values
        y = pairs["agent"].values

        # Compute linear fit and R^2
        try:
            slope, intercept = np.polyfit(x, y, 1)
            r = np.corrcoef(x, y)[0, 1]
            r2 = float(r * r)
        except Exception:
            slope, intercept, r, r2 = np.nan, np.nan, np.nan, np.nan

        # Identity line
        lo = min(np.min(x), np.min(y))
        hi = max(np.max(x), np.max(y))
        ax.plot([lo, hi], [lo, hi], 'k:', lw=1.2, label='identity')

        # Plot scatter - each point is one map's aggregated value
        scatter_label = f'{agg_label} per map'
        ax.scatter(x, y, s=16, alpha=0.65, color=SENSONAUT_COLOR, edgecolors='none', label=scatter_label)

        # Fit line
        if np.isfinite(slope):
            sign = '+' if intercept >= 0 else ''
            ax.plot([lo, hi], [slope * lo + intercept, slope * hi + intercept], color='#444', lw=1.6,
                    label=f'fit y={slope:.2f}x{sign}{intercept:.2f}, R²={r2:.2f}')
        ax.set_xlabel(f"Human {dv_label}", fontsize=Y_LABEL_FONTSIZE)
        ax.set_ylabel(f"Agent {dv_label}", fontsize=Y_LABEL_FONTSIZE)
        ax.tick_params(axis='both', labelsize=TICK_LABEL_FONTSIZE)

        # Legend in one row at top of each subplot
        ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.42), fontsize=LEGEND_FONTSIZE, frameon=True,
                  ncol=2, handlelength=1.0, alignment="center", handletextpad=0.4, columnspacing=-2.1)
        ax.grid(True, alpha=0.3)


    plt.tight_layout(rect=[0, 0, 1, 0.95])  # Leave room at top for legends
    plt.subplots_adjust(hspace=0.7)  # Reduce vertical gap between rows
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"[correlation board] Saved {out_path}")

def main():
    ap = argparse.ArgumentParser(description="Pruned plotting: unified main effects + correlation board.")
    ap.add_argument("--human-csv", default="analysis/df_human.csv",
                    help="Path to human data CSV")
    ap.add_argument("--mdf-glob", default="analysis/mdf_*.csv",
                    help="Glob pattern for model CSV files")
    ap.add_argument("--agents-dir", type=str, default=None,
                    help="Directory containing agent CSV files (overrides --mdf-glob)")
    ap.add_argument("--unified-main-effects-out", type=str, default="analysis/plots/unified_main_effects.png",
                    help="Output path for unified main effects figure")
    ap.add_argument("--correlation-board-out", type=str, default="analysis/plots/correlation_board.png",
                    help="Output path for correlation board figure")
    ap.add_argument("--corr-agg", type=str, default="median", choices=["median", "mean"],
                    help="Aggregation method for correlation plots")
    ap.add_argument("--duration-outlier-z", type=float, default=3.0,
                    help="Z threshold for removing duration outliers")
    ap.add_argument("--no-duration-outlier-filter", action="store_true",
                    help="Disable duration outlier filtering")

    args = ap.parse_args()

    # Determine model glob pattern
    mdf_glob = os.path.join(args.agents_dir, "mdf_*.csv") if args.agents_dir else args.mdf_glob

    # Load data
    human_df = load_human(args.human_csv)
    models = load_models(mdf_glob)

    print(f"Human data rows before outlier removal: {len(human_df)}")

    # Apply duration outlier filter
    if not args.no_duration_outlier_filter:
        human_df, (n_removed, pct_removed) = _filter_duration_outliers_global(
            human_df, z=args.duration_outlier_z, only_above=True
        )
        print(f"[duration outliers] Removed {n_removed} human trials ({pct_removed:.1f}%) using global {args.duration_outlier_z}σ rule.")

    if not models:
        print("No model CSV files found.")
        return

    print(f"Loaded {len(models)} model file(s)")

    # Export unified main effects
    export_unified_main_effects(human_df, models, args.unified_main_effects_out)
    
    # Note: You need to uncomment the body of export_correlation_board in your file
    export_correlation_board(human_df, models, args.correlation_board_out, agg=args.corr_agg)

    print("Done.")


if __name__ == "__main__":
    main()