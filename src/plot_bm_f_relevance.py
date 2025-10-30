# SHAP feature relevance bar chart with two flat colors (NO percent normalization)
# - Dataset features -> #e64042
# - Model features   -> #377eb8
# Sign/magnitude is shown only by bar length, not by color intensity.

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

# --- Data (from your logs) ---
scores = [0.0155387244321892, 0.08392635568362541, 0.01899605203507021,
          0.0171454101193659, 0.00842252436015099, 0.008961702337084203,
          0.004814474453121843, 0.009615550430965281, 0.031032761362213016,
          0.002765768355861129, 0.008468482690683347, 0.06255794064227714,
          0.052928236146963785, 0.07004764553426743, 0.005975406693164138,
          0.02202854974400945, 0.0067791090467023115, 0.042030186833070124,
          0.008279988113851494]

features = ['class_ent', 'eq_num_attr', 'gravity', 'inst_to_attr', 'nr_attr', 'nr_bin',
            'nr_class', 'nr_cor_attr', 'nr_inst', 'nr_norm', 'nr_outliers', 'ns_ratio',
            'Processing Units Number', 'Model Type', 'Training Complexity',
            'Prediction Complexity', 'Regularization', 'Robust to Outliers',
            'Representational Capacity']

# Optional: shorter labels for model-related features
rename_map = {
    'Processing Units Number': 'proc_units',
    'Model Type': 'model_type',
    'Training Complexity': 'train_complexity',
    'Prediction Complexity': 'inf_complexity',
    'Regularization': 'regularization',
    'Robust to Outliers': 'rob_outliers',
    'Representational Capacity': 'repr_capacity',
}

# --- Build dataframe (RAW values, no percent) ---
df = pd.DataFrame({'feature': features, 'relevance': scores})
df['feature'] = df['feature'].replace(rename_map)

# Define model vs dataset groups (after renaming)
MODEL_FEATURES = {
    'model_type', 'proc_units', 'train_complexity', 'inf_complexity',
    'regularization', 'rob_outliers', 'repr_capacity'
}
df['group'] = np.where(df['feature'].isin(MODEL_FEATURES), 'model', 'dataset')

# Sort ascending for horizontal bars
df = df.sort_values('relevance', ascending=True).reset_index(drop=True)

# --- Colors: just two flat colors ---
DATASET_COLOR = "#e64042"  # red
MODEL_COLOR = "#377eb8"    # blue

colors = [DATASET_COLOR if grp == 'dataset' else MODEL_COLOR for grp in df['group']]

# --- Plot style ---
plt.rcParams.update({
    "font.size": 12,
    "axes.titlesize": 14,
    "axes.labelsize": 13,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "legend.fontsize": 12,
    "figure.titlesize": 16
})

# --- Plot ---
fig, ax = plt.subplots(figsize=(6, 7))

ax.barh(df['feature'], df['relevance'], color=colors, edgecolor='none', linewidth=0)

# Labels & layout
ax.set_xlabel("Feature relevance (SHAP magnitude)")
ax.set_ylabel("Meta-features")

# X limits and light grid
xmax = float(df['relevance'].max())
ax.set_xlim(0, xmax * 1.18)
ax.grid(axis='x', linestyle='--', alpha=0.25)

# Clean look
for spine in ["top", "right", "left", "bottom"]:
    ax.spines[spine].set_visible(False)
ax.tick_params(axis='y', length=0)

# Legend with flat colors
legend_handles = [
    Patch(facecolor=DATASET_COLOR, edgecolor="none", label="Dataset features"),
    Patch(facecolor=MODEL_COLOR, edgecolor="none", label="Model features"),
]
ax.legend(handles=legend_handles, loc="lower right", frameon=False)

plt.tight_layout()

# Optional saves
plt.savefig("plots/pdf/f_relevance_shap.pdf", format='pdf', bbox_inches='tight')
plt.savefig("plots/png/f_relevance_shap.png", format='png', bbox_inches='tight')

plt.show()
