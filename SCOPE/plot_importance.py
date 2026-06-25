import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# Resolve script directory and paths
script_dir = os.path.dirname(os.path.abspath(__file__))
importance_csv_path = os.path.join(script_dir, "tabpfn_wavelength_importance.csv")

if not os.path.exists(importance_csv_path):
    raise FileNotFoundError(
        f"Wavelength importance CSV not found at: {importance_csv_path}\n"
        "Please run 'wavelenght_importance.py' once first to compute it."
    )

print(f"Loading wavelength importance from: {importance_csv_path}")
df_importance = pd.read_csv(importance_csv_path)

# Extract numeric wavelength from column names (e.g., R_550 -> 550.0)
try:
    df_importance['wavelength_numeric'] = df_importance['wavelength'].apply(
        lambda c: float(c.split("_")[1])
    )
except Exception as e:
    print(f"Warning parsing numeric wavelengths: {e}. Falling back to index-based ordering.")
    df_importance['wavelength_numeric'] = range(len(df_importance))

# Sort by numeric wavelength to ensure the line plot is drawn in correct spectral order
df_importance = df_importance.sort_values(by='wavelength_numeric').reset_index(drop=True)

# Masked regions (water band ranges)
WATER_BAND_RANGES = [
    (350.0, 400.0),
    (1310.0, 1470.0),
    (1750.0, 2010.0),
    (2390.0, 2500.0),
]

# Break the line at masked regions by inserting NaNs
x_sorted = df_importance['wavelength_numeric'].values
y_sorted = df_importance['importance'].values

x_plot = []
y_plot = []
for i in range(len(x_sorted)):
    x_plot.append(x_sorted[i])
    y_plot.append(y_sorted[i])
    if i < len(x_sorted) - 1:
        # Check if a water band falls between x_sorted[i] and x_sorted[i+1]
        for lo, hi in WATER_BAND_RANGES:
            if x_sorted[i] <= lo and x_sorted[i+1] >= hi:
                # Insert a NaN value at the midpoint of the gap to break the plot line
                x_plot.append((lo + hi) / 2.0)
                y_plot.append(np.nan)
                break

# Set style
sns.set_theme(style="whitegrid")

# Create figure
fig, ax = plt.subplots(figsize=(12, 5))

# Plot the relative importance line (using a single solid color)
ax.plot(
    x_plot, 
    y_plot, 
    linewidth=2,
    color='navy',
    zorder=2,
    label='Relative Importance'
)

# Add masked regions as grey vertical rectangles (opacity 30%)
first_mask = True
for lo, hi in WATER_BAND_RANGES:
    label = "Masked Regions" if first_mask else None
    ax.axvspan(lo, hi, color='grey', alpha=0.3, zorder=1, label=label)
    first_mask = False

# Add red vertical rectangle for ESA's FLEX Spectral Range (500 - 780 nm)
ax.axvspan(
    500.0, 780.0, 
    edgecolor='red', 
    facecolor='red', 
    alpha=0.15, 
    linestyle='--', 
    lw=1.5, 
    zorder=1, 
    label="ESA's FLEX Spectral Range"
)

# Axis limits
ax.set_xlim(x_sorted.min(), x_sorted.max())
ax.set_ylim(-0.02, 1.05)

# Axis labels
ax.set_xlabel("Wavelength (nm)", fontsize=12)
ax.set_ylabel("Relative importance", fontsize=12)

ax.grid(True, linestyle='--', alpha=0.6)
ax.legend(loc='upper right', frameon=True, facecolor='white', edgecolor='none')

plt.tight_layout()

# Save output plot
importance_plot_path = os.path.join(script_dir, "tabpfn_wavelength_importance.png")
plt.savefig(importance_plot_path, dpi=300)
plt.close()

print(f"✅ Saved importance plot to: {importance_plot_path}")
print("Done!")
