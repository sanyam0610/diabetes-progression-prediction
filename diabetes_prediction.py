"""
========================================================
 DIABETES PROGRESSION PREDICTION
 Author  : Sanyam Jain
 Tools   : Python | Scikit-learn | Pandas | NumPy | Matplotlib
 Dataset : Scikit-learn built-in Diabetes dataset (442 records)
========================================================

PROJECT OVERVIEW
----------------
Predicts diabetes disease progression one year after baseline
using patient health metrics: BMI, blood pressure, glucose levels etc.
Pipeline: Load → EDA → Feature Selection → Model Training → Evaluation → Visualisation
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.datasets import load_diabetes
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.feature_selection import f_regression
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# STEP 1 — LOAD DATASET
# ─────────────────────────────────────────────
diabetes = load_diabetes()
X = pd.DataFrame(diabetes.data, columns=diabetes.feature_names)
y = pd.Series(diabetes.target, name="progression")

print(f"Dataset shape : {X.shape}")
print(f"Target range  : {y.min():.0f} – {y.max():.0f}")
print(f"\nFeatures:\n{X.describe().round(2)}\n")

# ─────────────────────────────────────────────
# STEP 2 — EDA & OUTLIER TREATMENT
# ─────────────────────────────────────────────
"""
Outlier treatment using IQR method.
Outliers can skew Linear Regression significantly — removing them
improves model generalisation.
"""
df = X.copy()
df["progression"] = y

# IQR-based outlier removal
Q1 = df.quantile(0.25)
Q3 = df.quantile(0.75)
IQR = Q3 - Q1
before = len(df)
df = df[~((df < (Q1 - 1.5 * IQR)) | (df > (Q3 + 1.5 * IQR))).any(axis=1)]
after = len(df)
print(f"Outliers removed: {before - after} rows ({(before-after)/before*100:.1f}%)")

X_clean = df.drop("progression", axis=1)
y_clean = df["progression"]

# ─────────────────────────────────────────────
# STEP 3 — FEATURE SELECTION
# ─────────────────────────────────────────────
"""
Use F-statistic to rank features by their correlation with target.
Only keep features with p-value < 0.05 (statistically significant).
"""
f_stats, p_values = f_regression(X_clean, y_clean)
feature_scores = pd.DataFrame({
    "feature": X_clean.columns,
    "f_score": f_stats,
    "p_value": p_values
}).sort_values("f_score", ascending=False)

print("\nFeature Importance (F-score):")
print(feature_scores.to_string(index=False))

# Select significant features
selected = feature_scores[feature_scores["p_value"] < 0.05]["feature"].tolist()
print(f"\nSelected features: {selected}")
X_selected = X_clean[selected]

# ─────────────────────────────────────────────
# STEP 4 — TRAIN / TEST SPLIT & SCALING
# ─────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X_selected, y_clean, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc  = scaler.transform(X_test)

# ─────────────────────────────────────────────
# STEP 5 — MODEL TRAINING & EVALUATION
# ─────────────────────────────────────────────
# Baseline model (all features, no outlier removal)
X_base_train, X_base_test, y_base_train, y_base_test = train_test_split(
    X, y, test_size=0.2, random_state=42)
base_model = LinearRegression()
base_model.fit(X_base_train, y_base_train)
base_mse = mean_squared_error(y_base_test, base_model.predict(X_base_test))

# Optimised model
model = LinearRegression()
model.fit(X_train_sc, y_train)
y_pred = model.predict(X_test_sc)

mse      = mean_squared_error(y_test, y_pred)
rmse     = np.sqrt(mse)
mae      = mean_absolute_error(y_test, y_pred)
r2       = r2_score(y_test, y_pred)
mse_reduction = (base_mse - mse) / base_mse * 100

# Cross-validation
cv_scores = cross_val_score(model, X_train_sc, y_train, cv=5, scoring="r2")

print(f"\n{'='*45}")
print(f"  MODEL RESULTS")
print(f"{'='*45}")
print(f"  Baseline MSE      : {base_mse:.2f}")
print(f"  Optimised MSE     : {mse:.2f}")
print(f"  MSE Reduction     : {mse_reduction:.1f}%")
print(f"  RMSE              : {rmse:.2f}")
print(f"  MAE               : {mae:.2f}")
print(f"  R² Score          : {r2:.4f}")
print(f"  CV R² (mean±std)  : {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")
print(f"{'='*45}")

# ─────────────────────────────────────────────
# STEP 6 — VISUALISATION DASHBOARD
# ─────────────────────────────────────────────
BG     = "#f4f6fb"
NAVY   = "#1a237e"
ACCENT = "#3949ab"
ORANGE = "#ff6f00"

fig = plt.figure(figsize=(18, 12))
fig.patch.set_facecolor(BG)

fig.text(0.5, 0.97, "Diabetes Progression Prediction — Model Results",
         ha="center", va="top", fontsize=18, fontweight="bold", color=NAVY)
fig.text(0.5, 0.945,
         f"MSE Reduction: {mse_reduction:.1f}%  |  R² Score: {r2:.3f}  |  "
         f"RMSE: {rmse:.1f}  |  MAE: {mae:.1f}  |  CV R²: {cv_scores.mean():.3f}",
         ha="center", va="top", fontsize=10.5, color="#555")

gs = gridspec.GridSpec(2, 3, figure=fig,
                       hspace=0.42, wspace=0.35,
                       top=0.90, bottom=0.07, left=0.07, right=0.97)

def style_ax(ax, title):
    ax.set_facecolor("white")
    ax.set_title(title, fontsize=11, fontweight="bold", color=NAVY, pad=9)
    ax.tick_params(labelsize=9, colors="#444")
    for sp in ax.spines.values(): sp.set_edgecolor("#ddd")

# Panel 1 — Predicted vs Actual
ax1 = fig.add_subplot(gs[0, :2])
ax1.scatter(y_test, y_pred, color=ACCENT, alpha=0.6, s=50, edgecolors="white", linewidth=0.5)
lims = [min(y_test.min(), y_pred.min()) - 10, max(y_test.max(), y_pred.max()) + 10]
ax1.plot(lims, lims, "--", color=ORANGE, linewidth=2, label="Perfect prediction")
ax1.set_xlabel("Actual Progression", fontsize=9)
ax1.set_ylabel("Predicted Progression", fontsize=9)
ax1.legend(fontsize=9)
ax1.grid(color="#eee", linewidth=0.8)
style_ax(ax1, "📈  Predicted vs Actual Progression")
ax1.text(0.05, 0.92, f"R² = {r2:.3f}", transform=ax1.transAxes,
         fontsize=10, color=NAVY, fontweight="bold",
         bbox=dict(boxstyle="round,pad=0.3", facecolor="#E8EAF6", edgecolor=NAVY))

# Panel 2 — Feature Importance
ax2 = fig.add_subplot(gs[0, 2])
top_features = feature_scores.head(8)
colors2 = [NAVY if i < 3 else ACCENT for i in range(len(top_features))]
bars = ax2.barh(top_features["feature"], top_features["f_score"],
                color=colors2, edgecolor="white", height=0.6)
style_ax(ax2, "🏷  Feature Importance (F-score)")
ax2.set_xlabel("F-score", fontsize=9)
ax2.grid(axis="x", color="#eee"); ax2.grid(axis="y", visible=False)
for bar, val in zip(bars, top_features["f_score"]):
    ax2.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2,
             f"{val:.1f}", va="center", fontsize=8)

# Panel 3 — Residuals
ax3 = fig.add_subplot(gs[1, 0])
residuals = y_test.values - y_pred
ax3.scatter(y_pred, residuals, color=ACCENT, alpha=0.6, s=40, edgecolors="white")
ax3.axhline(0, color=ORANGE, linewidth=2, linestyle="--")
ax3.set_xlabel("Predicted Values", fontsize=9)
ax3.set_ylabel("Residuals", fontsize=9)
ax3.grid(color="#eee", linewidth=0.8)
style_ax(ax3, "📉  Residual Plot")

# Panel 4 — MSE Comparison
ax4 = fig.add_subplot(gs[1, 1])
bars4 = ax4.bar(["Baseline\n(No Optimisation)", "Optimised\n(Feature Selection\n+ Outlier Removal)"],
                [base_mse, mse], color=[ACCENT, NAVY], edgecolor="white", width=0.5)
ax4.set_ylabel("MSE", fontsize=9)
style_ax(ax4, f"🎯  MSE Reduction: {mse_reduction:.1f}%")
ax4.grid(axis="y", color="#eee")
for bar, val in zip(bars4, [base_mse, mse]):
    ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 50,
             f"{val:.0f}", ha="center", fontsize=9, fontweight="bold", color="#333")
ax4.annotate("", xy=(1, mse + 200), xytext=(0, base_mse - 200),
             arrowprops=dict(arrowstyle="->", color=ORANGE, lw=2))
ax4.text(0.5, (base_mse + mse) / 2, f"↓ {mse_reduction:.1f}%",
         ha="center", fontsize=10, color=ORANGE, fontweight="bold", transform=ax4.get_yaxis_transform())

# Panel 5 — Cross Validation
ax5 = fig.add_subplot(gs[1, 2])
folds = [f"Fold {i+1}" for i in range(5)]
bar_colors = [NAVY if s == cv_scores.max() else ACCENT for s in cv_scores]
ax5.bar(folds, cv_scores, color=bar_colors, edgecolor="white", width=0.55)
ax5.axhline(cv_scores.mean(), color=ORANGE, linewidth=2,
            linestyle="--", label=f"Mean R²={cv_scores.mean():.3f}")
ax5.set_ylabel("R² Score", fontsize=9)
ax5.legend(fontsize=8)
style_ax(ax5, "🔁  5-Fold Cross Validation")
ax5.grid(axis="y", color="#eee")

out = "/mnt/user-data/outputs/diabetes_prediction_results.png"
plt.savefig(out, dpi=150, bbox_inches="tight", facecolor=BG)
plt.close()
print(f"\n✅ Dashboard saved → {out}")
print("🎉 Project complete!")
