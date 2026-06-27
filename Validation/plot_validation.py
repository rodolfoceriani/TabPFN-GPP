#!/usr/bin/env python3
"""
plot_predictions.py

"""

import os
import math
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error

# ── Paths ────────────────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(SCRIPT_DIR, "validation.csv")
OUT_DIR = SCRIPT_DIR

# ── Helper: Calculate Pearson r ──────────────────────────────────────
def pearson_r(y_t, y_p):
    y_t = np.asarray(y_t)
    y_p = np.asarray(y_p)
    if len(y_t) < 2 or np.std(y_t) == 0 or np.std(y_p) == 0:
        return 0.0
    return np.corrcoef(y_t, y_p)[0, 1]

def main():
    if not os.path.exists(CSV_PATH):
        print(f"❌ Error: Predictions CSV not found at '{CSV_PATH}'")
        return

    print(f"📖 Loading predictions from: {CSV_PATH}")
    df = pd.read_csv(CSV_PATH)
    df['landcover'] = df['landcover'].apply(lambda x: str(x).capitalize())
    print(f"Loaded {len(df)} rows × {df.shape[1]} columns")

    # Extract target values
    y = df['GPP_observed'].values
    y_pred = df['GPP_predicted'].values
    landcover = df['landcover'].values
    network = df['network'].values

    # Calculate overall metrics
    r_val = pearson_r(y, y_pred)
    r2    = r_val ** 2
    rmse  = math.sqrt(mean_squared_error(y, y_pred))
    rrmse = rmse / np.mean(y) * 100  # relative RMSE in %

    print(f"\n── Validation metrics ──")
    print(f"r     = {r_val:.4f}")
    print(f"R²    = {r2:.4f}")
    print(f"RMSE  = {rmse:.4f}")
    print(f"rRMSE = {rrmse:.2f} %")

    print("\n── Sample counts per Land Cover ──")
    print(df['landcover'].value_counts().to_string())

    print("\n── Sample counts per Network ──")
    print(df['network'].value_counts().to_string())

    # Define plot limits and margins
    LOWER_LIMIT = min(y.min(), y_pred.min())
    UPPER_LIMIT = max(y.max(), y_pred.max())
    margin = (UPPER_LIMIT - LOWER_LIMIT) * 0.05
    LOWER_LIMIT -= margin
    UPPER_LIMIT += margin

    textstr = f'$r = {r_val:.2f}$\n$R^2 = {r2:.2f}$\n$RMSE = {rmse:.2f}$\n$rRMSE = {rrmse:.2f}\\%$'

    # ── Plot 1: LandCover scatter plot ───────────────────────────────
    # Calculate Pearson r separately for each LandCover
    unique_covers = df['landcover'].dropna().unique()
    lc_r_dict = {}
    for cover in unique_covers:
        cover_df = df[df['landcover'] == cover]
        lc_r_dict[cover] = pearson_r(cover_df['GPP_observed'].values, cover_df['GPP_predicted'].values)

    # Map LandCover to labels including their respective Pearson r and R²
    df['LandCover_Legend'] = df['landcover'].map(
        lambda x: f"{x} ($r={lc_r_dict[x]:.2f}, R^2 = {lc_r_dict[x]**2:.2f}$)" if x in lc_r_dict else x
    )

    # Define custom ordering for landcover legend: Cropland first, Forest second, then others
    ordered_covers = []
    preferred_order = ['Cropland', 'Forest', 'Grassland', 'Wetland']
    for cover in preferred_order:
        if cover in lc_r_dict:
            ordered_covers.append(cover)
    for cover in lc_r_dict:
        if cover not in ordered_covers:
            ordered_covers.append(cover)

    hue_order = [
        f"{cover} ($r={lc_r_dict[cover]:.2f}, R^2 = {lc_r_dict[cover]**2:.2f}$)"
        for cover in ordered_covers
    ]

    # Map Forest to black and others to Set1 colors
    base_colors = sns.color_palette('Set1', len(hue_order))
    custom_palette = {}
    color_idx = 0
    for label in hue_order:
        if 'Forest' in label:
            custom_palette[label] = 'black'
        else:
            custom_palette[label] = base_colors[color_idx]
            color_idx += 1

    print("Generating LandCover scatterplot...")
    plt.figure(figsize=(10, 8))
    sns.scatterplot(
        data=df,
        x='GPP_observed',
        y='GPP_predicted',
        hue='LandCover_Legend',
        hue_order=hue_order,
        palette=custom_palette,
        s=40,
        alpha=0.8
    )

    plt.plot([LOWER_LIMIT, UPPER_LIMIT], [LOWER_LIMIT, UPPER_LIMIT], 'r--', lw=2.5, label='1:1 line')
    sns.regplot(x=df['GPP_observed'], y=df['GPP_predicted'], scatter=False,
                color='darkblue', line_kws={'lw': 3, 'label': 'Model Fit'}, ci=None)

    plt.xlabel(r'Observed GPP [$\mu$mol CO$_2$ m$^{-2}$ s$^{-1}$]', fontsize=14, fontweight='bold')
    plt.ylabel(r'Predicted GPP [$\mu$mol CO$_2$ m$^{-2}$ s$^{-1}$]', fontsize=14, fontweight='bold')
    #plt.title(f'Independent Validation (n = {len(y)})', fontsize=16, fontweight='bold', pad=20)

    plt.text(0.05, 0.95, textstr, transform=plt.gca().transAxes, fontsize=14,
             verticalalignment='top', bbox=dict(boxstyle='round,pad=0.5', facecolor='white', edgecolor='lightgray', alpha=0.9))

    plt.legend(loc='lower right', fontsize=12, frameon=True, shadow=True)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.xlim(LOWER_LIMIT, UPPER_LIMIT)
    plt.ylim(LOWER_LIMIT, UPPER_LIMIT)
    plt.gca().set_aspect('equal', adjustable='box')

    plt.tight_layout()
    lc_plot_path = os.path.join(OUT_DIR, "validation_landcover.png")
    plt.savefig(lc_plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Saved LandCover scatter plot → {lc_plot_path}")

    # ── Plot 2: Network scatter plot ─────────────────────────────────
    # Calculate Pearson r separately for each network
    unique_networks = df['network'].dropna().unique()
    ds_r_dict = {}
    for ds in unique_networks:
        ds_df = df[df['network'] == ds]
        ds_r_dict[ds] = pearson_r(ds_df['GPP_observed'].values, ds_df['GPP_predicted'].values)

    # Map network to labels including their respective Pearson r and R²
    df['Dataset_Legend'] = df['network'].map(
        lambda x: f"{x} ($r={ds_r_dict[x]:.2f}, R^2 = {ds_r_dict[x]**2:.2f}$)" if x in ds_r_dict else x
    )

    print("Generating Network scatterplot...")
    plt.figure(figsize=(10, 8))
    sns.scatterplot(
        data=df,
        x='GPP_observed',
        y='GPP_predicted',
        hue='Dataset_Legend',
        palette='Dark2',
        s=45,
        alpha=0.8
    )

    plt.plot([LOWER_LIMIT, UPPER_LIMIT], [LOWER_LIMIT, UPPER_LIMIT], 'r--', lw=2.5, label='1:1 line')
    sns.regplot(x=df['GPP_observed'], y=df['GPP_predicted'], scatter=False,
                color='darkblue', line_kws={'lw': 3, 'label': 'Model Fit'}, ci=None)

    plt.xlabel(r'Observed GPP [$\mu$mol CO$_2$ m$^{-2}$ s$^{-1}$]', fontsize=14, fontweight='bold')
    plt.ylabel(r'Predicted GPP [$\mu$mol CO$_2$ m$^{-2}$ s$^{-1}$]', fontsize=14, fontweight='bold')
    #plt.title(f'Independent Validation (n = {len(y)})', fontsize=16, fontweight='bold', pad=20)

    plt.text(0.05, 0.95, textstr, transform=plt.gca().transAxes, fontsize=14,
             verticalalignment='top', bbox=dict(boxstyle='round,pad=0.5', facecolor='white', edgecolor='lightgray', alpha=0.9))

    plt.legend(loc='lower right', fontsize=12, frameon=True, shadow=True)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.xlim(LOWER_LIMIT, UPPER_LIMIT)
    plt.ylim(LOWER_LIMIT, UPPER_LIMIT)
    plt.gca().set_aspect('equal', adjustable='box')

    plt.tight_layout()
    ds_plot_path = os.path.join(OUT_DIR, "validation_dataset.png")
    plt.savefig(ds_plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Saved Dataset scatter plot → {ds_plot_path}")

if __name__ == "__main__":
    main()
