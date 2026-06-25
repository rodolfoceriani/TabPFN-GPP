import os
import math
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import torch

from scipy.stats import gaussian_kde
from sklearn.model_selection import train_test_split
from sklearn.cross_decomposition import PLSRegression
from sklearn.metrics import r2_score, mean_squared_error
from tabpfn import TabPFNRegressor
import joblib


# ==========================================
# 1. CONFIGURATION & PATH RESOLUTION
# ==========================================

SEED = 42

# Number of PLS components for spectral data
N_PLS_COMPONENTS = 20

# Input variables besides spectra
OTHER_COLS = ['Ta', 'Rin', 'ea', 'SMC']

# Target variable
TARGET_COL = 'GPP'

# Bad/water/noisy spectral regions
WATER_BAND_RANGES = [
    (350.0, 400.0),
    (1310.0, 1470.0),
    (1750.0, 2010.0),
    (2390.0, 2500.0),
]

# Resolve script directory
script_dir = os.path.dirname(os.path.abspath(__file__))

# Try local directory first, then relative path fallback
CSV_PATH = os.path.join(script_dir, "dataset_GPP_Ta_Rin_ea_reflectance_EMIT.csv")

if not os.path.exists(CSV_PATH):
    CSV_PATH = os.path.abspath(
        os.path.join(
            script_dir,
            "..",
            "output",
            "verification_run_2026-06-07-2138",
            "dataset_GPP_Ta_Rin_ea_reflectance_EMIT.csv"
        )
    )

if not os.path.exists(CSV_PATH):
    CSV_PATH = "output/verification_run_2026-06-07-2138/dataset_GPP_Ta_Rin_ea_reflectance_EMIT.csv"

print(f"Loading dataset from: {CSV_PATH}")


# ==========================================
# 2. DATA LOADING & BAD BANDS REMOVAL
# ==========================================

df = pd.read_csv(CSV_PATH)
print(f"Original dataset shape: {df.shape}")


def drop_water_bands(df, water_ranges):
    """
    Drop reflectance columns R_<wavelength> that fall inside specified bad-band ranges.
    """
    refl_cols = [c for c in df.columns if c.startswith("R_")]
    cols_to_drop = []

    for col in refl_cols:
        try:
            wl = float(col.replace("R_", ""))
        except ValueError:
            continue

        for lo, hi in water_ranges:
            if lo <= wl <= hi:
                cols_to_drop.append(col)
                break

    df_dropped = df.drop(columns=cols_to_drop)

    remaining_refl_cols = [c for c in df_dropped.columns if c.startswith("R_")]

    print(f"  Dropped {len(cols_to_drop)} water/noisy bands.")
    print(f"  Remaining reflectance bands: {len(remaining_refl_cols)}")

    return df_dropped


df = drop_water_bands(df, WATER_BAND_RANGES)


# ==========================================
# 3. DATA VALIDATION & CLEANING
# ==========================================

# Check required columns
required_cols = OTHER_COLS + [TARGET_COL]
missing_required = [c for c in required_cols if c not in df.columns]

if missing_required:
    raise ValueError(f"Missing required columns in dataset: {missing_required}")

# Get reflectance columns after bad-band removal
refl_cols = [c for c in df.columns if c.startswith("R_")]

if len(refl_cols) == 0:
    raise ValueError("No reflectance columns found. Expected columns starting with 'R_'.")

# Convert relevant columns to numeric, coercing problematic values to NaN
cols_for_model = OTHER_COLS + [TARGET_COL] + refl_cols
df[cols_for_model] = df[cols_for_model].apply(pd.to_numeric, errors='coerce')

initial_rows = len(df)
df = df.dropna(subset=cols_for_model).reset_index(drop=True)

print(f"Data cleaning: discarded {initial_rows - len(df)} rows with NA/non-numeric values.")
print(f"Final cleaned dataset shape: {df.shape}")

if len(df) < 10:
    raise ValueError("Very few rows remain after cleaning. Check dataset and bad-band filtering.")


# ==========================================
# 4. TRAIN / TEST SPLIT
# ==========================================

train_df, test_df = train_test_split(
    df,
    test_size=0.30,
    random_state=SEED
)

print(f"Dataset split: Train={len(train_df)} rows, Test={len(test_df)} rows")


# ==========================================
# 5. PREPARE SPECTRAL DATA AND TARGET
# ==========================================

X_train_spectral = train_df[refl_cols].values
X_test_spectral = test_df[refl_cols].values

X_train_other = train_df[OTHER_COLS].values
X_test_other = test_df[OTHER_COLS].values

y_train = train_df[TARGET_COL].values
y_test = test_df[TARGET_COL].values

# PLS components cannot exceed min(n_samples - 1, n_features)
max_pls_components = min(X_train_spectral.shape[0] - 1, X_train_spectral.shape[1])

if N_PLS_COMPONENTS > max_pls_components:
    print(
        f"Requested N_PLS_COMPONENTS={N_PLS_COMPONENTS}, "
        f"but maximum allowed is {max_pls_components}. "
        f"Using {max_pls_components} instead."
    )
    N_PLS_COMPONENTS = max_pls_components

if N_PLS_COMPONENTS < 1:
    raise ValueError("N_PLS_COMPONENTS must be at least 1.")


# ==========================================
# 6. PLS ON SPECTRAL DATA
#    Fit only on training data to avoid leakage
# ==========================================

print(f"Fitting PLS with {N_PLS_COMPONENTS} components on training spectral data...")

pls = PLSRegression(
    n_components=N_PLS_COMPONENTS,
    scale=True
)

# fit_transform returns a tuple: X_scores, Y_scores
X_train_pls = pls.fit_transform(X_train_spectral, y_train)[0]
X_test_pls = pls.transform(X_test_spectral)

pls_cols = [f"PLS_{i + 1}" for i in range(N_PLS_COMPONENTS)]

print(f"PLS-transformed train shape: {X_train_pls.shape}")
print(f"PLS-transformed test shape: {X_test_pls.shape}")


# ==========================================
# 7. ASSEMBLE TABPFN MODEL INPUTS
# ==========================================

X_train = np.hstack([X_train_other, X_train_pls])
X_test = np.hstack([X_test_other, X_test_pls])

feature_names = OTHER_COLS + pls_cols

print(f"Final TabPFN train feature matrix shape: {X_train.shape}")
print(f"Final TabPFN test feature matrix shape: {X_test.shape}")
print(f"Number of TabPFN input features: {len(feature_names)}")


# ==========================================
# 8. TRAIN TABPFN REGRESSOR
# ==========================================

print("Initializing TabPFNRegressor...")

# Auto-detect device (GPU/CPU) for TabPFN compliance
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")
model = TabPFNRegressor(device=device)

print("Training TabPFNRegressor model...")
model.fit(X_train, y_train)

print("Predicting on independent test set...")
y_pred = model.predict(X_test)


# ==========================================
# 9. SAVE PREDICTIONS
# ==========================================

predictions_df = pd.DataFrame({
    'observed': y_test,
    'predicted': y_pred
})

predictions_path = os.path.join(script_dir, "GPP_test_predictions_TabPFN_PLS.csv")
predictions_df.to_csv(predictions_path, index=False)

print(f"✅ Saved test predictions to: {predictions_path}")


# ==========================================
# 10. SAVE MODEL, PLS OBJECT, AND FEATURE INFO
# ==========================================

model_path = os.path.join(script_dir, "tabpfn_model_PLS.joblib")
pls_path = os.path.join(script_dir, "pls_model_PLS.joblib")
feature_info_path = os.path.join(script_dir, "tabpfn_pls_feature_info.joblib")

joblib.dump(model, model_path)
joblib.dump(pls, pls_path)

feature_info = {
    "other_cols": OTHER_COLS,
    "target_col": TARGET_COL,
    "refl_cols": refl_cols,
    "pls_cols": pls_cols,
    "feature_names": feature_names,
    "n_pls_components": N_PLS_COMPONENTS,
    "water_band_ranges": WATER_BAND_RANGES,
}

joblib.dump(feature_info, feature_info_path)

print(f"✅ Saved TabPFN model to: {model_path}")
print(f"✅ Saved PLS object to: {pls_path}")
print(f"✅ Saved feature info to: {feature_info_path}")


# ==========================================
# 11. PERFORMANCE METRICS
# ==========================================

r2 = r2_score(y_test, y_pred)
rmse = math.sqrt(mean_squared_error(y_test, y_pred))

mean_obs = np.mean(y_test)
rrmse = (rmse / mean_obs) * 100 if mean_obs != 0 else np.nan

print("=" * 60)
print("Test Set Metrics")
print("=" * 60)
print(f"R²    : {r2:.4f}")
print(f"RMSE  : {rmse:.4f}")
print(f"rRMSE : {rrmse:.2f}%")
print("=" * 60)


# ==========================================
# 12. DENSITY SCATTERPLOT
# ==========================================

# Calculate point density safely
xy = np.vstack([y_test, y_pred])

try:
    z = gaussian_kde(xy)(xy)
except Exception as e:
    print(f"Warning: gaussian_kde failed due to: {e}")
    print("Using uniform density values instead.")
    z = np.ones_like(y_test, dtype=float)

# Sort points by density, so densest points are plotted last
idx = z.argsort()
x_plot = y_test[idx]
y_plot = y_pred[idx]
z_plot = z[idx]

plt.figure(figsize=(10, 8))

scatter = plt.scatter(
    x_plot,
    y_plot,
    c=z_plot,
    s=20,
    cmap='jet',
    edgecolor='none'
)

cb = plt.colorbar(scatter)
cb.set_label('Density', fontsize=12)

if np.nanmin(z_plot) != np.nanmax(z_plot):
    cb.set_ticks([z_plot.min(), z_plot.max()])
    cb.set_ticklabels(['Low', 'High'])

# Dynamic limits based on observed and predicted values
LOWER_LIMIT = min(np.min(y_test), np.min(y_pred))
UPPER_LIMIT = max(np.max(y_test), np.max(y_pred))

margin = (UPPER_LIMIT - LOWER_LIMIT) * 0.05

if margin == 0:
    margin = 1.0

LOWER_LIMIT -= margin
UPPER_LIMIT += margin

# 1:1 line
plt.plot(
    [LOWER_LIMIT, UPPER_LIMIT],
    [LOWER_LIMIT, UPPER_LIMIT],
    'r--',
    lw=2.5,
    label='1:1 line'
)

# Regression line
sns.regplot(
    x=predictions_df['observed'],
    y=predictions_df['predicted'],
    scatter=False,
    color='darkblue',
    line_kws={'lw': 3, 'label': 'Linear fit'},
    ci=None
)

plt.xlabel('Observed GPP', fontsize=14, fontweight='bold')
plt.ylabel('Predicted GPP', fontsize=14, fontweight='bold')
plt.title(
    'GPP Observed vs. Predicted — TabPFN Regressor with PLS Features',
    fontsize=16,
    fontweight='bold',
    pad=20
)

# Metrics box
textstr = (
    f"$R^2$ = {r2:.3f}\n"
    f"RMSE = {rmse:.3f}\n"
    f"rRMSE = {rrmse:.2f}%"
)

plt.text(
    0.05,
    0.95,
    textstr,
    transform=plt.gca().transAxes,
    fontsize=14,
    verticalalignment='top',
    bbox=dict(
        boxstyle='round,pad=0.5',
        facecolor='white',
        edgecolor='lightgray',
        alpha=0.9
    )
)

plt.legend(loc='lower right', fontsize=12, frameon=True, shadow=True)
plt.grid(True, linestyle='--', alpha=0.7)
plt.axis('equal')
plt.xlim(LOWER_LIMIT, UPPER_LIMIT)
plt.ylim(LOWER_LIMIT, UPPER_LIMIT)

plt.tight_layout()

plot_path = os.path.join(script_dir, "GPP_test_density_scatter_TabPFN_PLS.png")
plt.savefig(plot_path, dpi=300, bbox_inches='tight')
plt.close()

print(f"✅ Density scatterplot saved to: {plot_path}")


# ==========================================
# 13. OPTIONAL: SAVE PLS SCORES FOR INSPECTION
# ==========================================

train_scores_df = pd.DataFrame(X_train_pls, columns=pls_cols)
test_scores_df = pd.DataFrame(X_test_pls, columns=pls_cols)

train_scores_df[TARGET_COL] = y_train
test_scores_df[TARGET_COL] = y_test

train_scores_path = os.path.join(script_dir, "train_PLS_scores.csv")
test_scores_path = os.path.join(script_dir, "test_PLS_scores.csv")

train_scores_df.to_csv(train_scores_path, index=False)
test_scores_df.to_csv(test_scores_path, index=False)

print(f"✅ Saved train PLS scores to: {train_scores_path}")
print(f"✅ Saved test PLS scores to: {test_scores_path}")

print("Done.")
