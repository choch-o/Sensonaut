#!/bin/bash
# Run test and analysis for a given model

# Define model name 
model_name="test_sensonaut_chi2026" # CHI 2026 VERSION

df_human="analysis/data/df_human.csv"

# Step 1: Run test script (dry run mode)
python scripts/test_models_and_analyze.py \
  --dry-run \
  --model-name "$model_name" \
  --human-csv "$df_human" 

# Expected outputs from Step 1:
# analysis/model-$model_name/mdf_$model_name.csv
# analysis/model-$model_name/mdf_models_vs_humans_summary.csv

# Step 2: Plot results
python analysis/plot_models_vs_humans_facets.py \
  --human-csv "$df_human" \
  --mdf-glob "analysis/model-$model_name/mdf_*.csv" \
  --unified-main-effects-out analysis/plots/unified_main_effects.png \
  --correlation-board-out analysis/plots/correlation_board.png \
  --corr-agg median
  --export-sorted-media \
  --agents-dir "analysis/model-$model_name" \
  --compact \
  --models-out-dir "analysis/model-$model_name" \
  --export-unified-main-effects \
  --export-corr \
  --corr-agg median \
  --corr-out "analysis/model-$model_name/corr_by_{dv}.png" \
  --export-loo \
  --loo-out "analysis/model-$model_name/loo_{dv}.csv" \
  --export-accuracy \
  --accuracy-out "analysis/model-$model_name/accuracy_{dv}.csv" \
  --export-accuracy-bars \
  --accbar-out "analysis/model-$model_name/accuracy_bars.png" \
  --accbar-level both \
  --export-overall-bars \
  --overall-out "analysis/model-$model_name/overall_bars.png" \
  --export-rank-corr \
  --rank-agg median \
  --rank-out "analysis/model-$model_name/rankcorr_{dv}_{subset}.png" \
  --rank-csv "analysis/model-$model_name/rankcorr_summary_{dv}.csv" \
  --export-accuracy-boxes \
  --accbox-out "analysis/model-$model_name/accuracy_boxes.png" \
  --accbox-level both  


  # Step 3: Plot patterns - action histogram
  python analysis/compare_traj.py \
    --model-name "$model_name" \

  # Step 4: Belief action comparison
  python analysis/belief_action_comparison.py \
    --human-root "analysis/data" \
    --pid "p05" \
    --agent-root "test_logs/$model_name" \
    --agent-id "agent_4" \
    --map-id "M0175" \
    --audio-like-source "geometric" \
    --agent-belief "saved" 

