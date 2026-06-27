#!/usr/bin/env python3
"""
plot_comparison.py
------------------
Produces a 1×3 joint figure comparing EMIT GPP, MODIS GPP, and GOSIF GPP
against daily eddy covariance tower measurements, colored by land cover.

Layout:  EMIT  |  MODIS  |  GOSIF

Loads and plots data from gpp_comparison.csv.
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
INPUT_CSV = os.path.join(SCRIPT_DIR, 'gpp_comparison.csv')
OUTPUT_PLOT = os.path.join(SCRIPT_DIR, 'gpp_comparison_joint.png')


# ── Helpers ──────────────────────────────────────────────────────────
def pearson_r(y_t, y_p):
    """Calculate Pearson correlation coefficient."""
    y_t = np.asarray(y_t, dtype=float)
    y_p = np.asarray(y_p, dtype=float)
    if len(y_t) < 2 or np.std(y_t) == 0 or np.std(y_p) == 0:
        return 0.0
    return np.corrcoef(y_t, y_p)[0, 1]


# ── Main ─────────────────────────────────────────────────────────────
def main():
    # ── 1. Load the CSV dataset ──────────────────────────────────────
    print("📖 Loading GPP comparison dataset...")
    if not os.path.exists(INPUT_CSV):
        print(f"❌ Error: {INPUT_CSV} not found.")
        return
    valid_df = pd.read_csv(INPUT_CSV)
    print(f"   Loaded {len(valid_df)} observations")

    # Clean / handle missing values if any
    valid_df = valid_df.dropna(subset=['OBSERVED', 'EMIT', 'MODIS', 'GOSIF', 'LANDCOVER']).copy()
    print(f"   Valid observations after dropping NaNs: {len(valid_df)}")

    if len(valid_df) == 0:
        print("❌ Error: No valid observations found.")
        return

    print("\n── Observations per Land Cover ──")
    for lc, count in valid_df['LANDCOVER'].value_counts().items():
        print(f"   {str(lc).capitalize():>12s}: {count}")

    # ── 2. Generate joint figure ─────────────────────────────────────
    print("\n🎨 Generating joint comparison figure...")

    # Capitalize landcover for consistent display
    valid_df['LANDCOVER'] = valid_df['LANDCOVER'].apply(
        lambda x: str(x).capitalize()
    )

    # Panel definitions
    panels = [
        {
            'title': 'EMIT',
            'y_col': 'EMIT',
            'y_label': r'EMIT GPP ($g\ C\ m^{-2}\ d^{-1}$)',
        },
        {
            'title': 'MODIS',
            'y_col': 'MODIS',
            'y_label': r'MODIS GPP ($g\ C\ m^{-2}\ d^{-1}$)',
        },
        {
            'title': 'GOSIF',
            'y_col': 'GOSIF',
            'y_label': r'GOSIF GPP ($g\ C\ m^{-2}\ d^{-1}$)',
        },
    ]

    # Calculate global axis limits across all panels
    all_vals = np.concatenate([
        valid_df['OBSERVED'].values,
        valid_df['EMIT'].values,
        valid_df['MODIS'].values,
        valid_df['GOSIF'].values,
    ])
    LOWER_LIMIT = -0.5
    UPPER_LIMIT = all_vals.max()
    margin = (UPPER_LIMIT - LOWER_LIMIT) * 0.05
    UPPER_LIMIT += margin

    fig, axes = plt.subplots(1, 3, figsize=(24, 8))

    for ax, panel in zip(axes, panels):
        y_obs = valid_df['OBSERVED'].values
        y_pred = valid_df[panel['y_col']].values

        # ── Overall metrics ──
        r_val = pearson_r(y_obs, y_pred)
        r2 = r_val ** 2
        rmse = math.sqrt(mean_squared_error(y_obs, y_pred))
        rrmse = rmse / np.mean(y_obs) * 100

        # ── Per-landcover Pearson r for legend ──
        unique_covers = sorted(valid_df['LANDCOVER'].dropna().unique())
        lc_r_dict = {}
        for cover in unique_covers:
            cover_mask = valid_df['LANDCOVER'] == cover
            lc_r_dict[cover] = pearson_r(
                valid_df.loc[cover_mask, 'OBSERVED'].values,
                valid_df.loc[cover_mask, panel['y_col']].values,
            )

        # Build legend labels
        plot_df = valid_df.copy()
        plot_df['LandCover_Legend'] = plot_df['LANDCOVER'].map(
            lambda x: (
                f"{x} ($R^2={lc_r_dict[x]**2:.2f}$)"
            )
            if x in lc_r_dict
            else x
        )

        hue_order = [
            f"{cover} ($R^2={lc_r_dict[cover]**2:.2f}$)"
            for cover in unique_covers if cover in lc_r_dict
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

        # Scatter by landcover
        sns.scatterplot(
            data=plot_df,
            x='OBSERVED',
            y=panel['y_col'],
            hue='LandCover_Legend',
            hue_order=hue_order,
            palette=custom_palette,
            s=40,
            alpha=0.8,
            ax=ax,
        )

        # 1:1 line
        ax.plot(
            [LOWER_LIMIT, UPPER_LIMIT],
            [LOWER_LIMIT, UPPER_LIMIT],
            'r--', lw=2.5, label='1:1 line',
        )

        # Regression fit
        sns.regplot(
            x=valid_df['OBSERVED'],
            y=valid_df[panel['y_col']],
            scatter=False,
            color='darkblue',
            line_kws={'lw': 3, 'label': 'Linear Fit'},
            ci=None,
            ax=ax,
        )

        # Axis labels & title
        ax.set_xlabel(
            r'Observed Daily GPP (Tower, $g\ C\ m^{-2}\ d^{-1}$)',
            fontsize=16.9, fontweight='bold',
        )
        ax.set_ylabel(panel['y_label'], fontsize=16.9, fontweight='bold')
        ax.set_title(
            panel['title'],
            fontsize=19.5, fontweight='bold', pad=15,
        )

        # Metrics text box
        textstr = (
            f'$R^2 = {r2:.2f}$\n'
            f'$RMSE = {rmse:.2f}$\n'
            f'$rRMSE = {rrmse:.2f}\\%$'
        )
        ax.text(
            0.05, 0.95, textstr,
            transform=ax.transAxes, fontsize=19.5,
            verticalalignment='top',
            bbox=dict(
                boxstyle='round,pad=0.5',
                facecolor='white',
                edgecolor='lightgray',
                alpha=0.9,
            ),
        )

        # Formatting
        ax.legend(loc='upper right', fontsize=16.9, frameon=True, shadow=True)
        ax.tick_params(axis='both', which='major', labelsize=13)
        ax.grid(True, linestyle='--', alpha=0.7)
        ax.set_xlim(LOWER_LIMIT, UPPER_LIMIT)
        ax.set_ylim(LOWER_LIMIT, UPPER_LIMIT)
        ax.set_aspect('equal', adjustable='box')

    plt.tight_layout()

    os.makedirs(SCRIPT_DIR, exist_ok=True)
    plt.savefig(OUTPUT_PLOT, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"\n✅ Saved joint comparison figure → {OUTPUT_PLOT}")

    # ── Print summary ────────────────────────────────────────────────
    print(f"\n── Summary ──")
    print(f"Total observations: {len(valid_df)}")
    print(f"Land covers: {', '.join(sorted(valid_df['LANDCOVER'].unique()))}")

    for panel in panels:
        y_obs = valid_df['OBSERVED'].values
        y_pred = valid_df[panel['y_col']].values
        r_val = pearson_r(y_obs, y_pred)
        rmse = math.sqrt(mean_squared_error(y_obs, y_pred))
        print(
            f"  {panel['title']:>12s}:  r={r_val:.4f}  "
            f"R²={r_val**2:.4f}  RMSE={rmse:.4f}"
        )


if __name__ == '__main__':
    main()
