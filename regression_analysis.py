"""
Part 1b: Multiple Linear Regression
What audio features are associated with Spotify popularity?

Model: popularity ~ danceability + energy + valence + tempo + acousticness
                     + loudness + instrumentalness + speechiness

Includes:
- Coefficients, standard errors, t-stats, p-values, 95% confidence intervals
- R-squared (train) and R-squared on a held-out test set (generalization check)
- An explicit causality disclaimer

Run locally:
    pip3 install pandas numpy scipy scikit-learn
    python3 regression_analysis.py
"""

import pandas as pd
import numpy as np
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

df = pd.read_csv("dataset_cleaned.csv")

features = ["danceability", "energy", "valence", "tempo", "acousticness",
            "loudness", "instrumentalness", "speechiness"]
X = df[features].values
y = df["popularity"].values

# ---------- Train/test split (80/20) to check generalization ----------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LinearRegression()
model.fit(X_train, y_train)

train_r2 = model.score(X_train, y_train)
test_pred = model.predict(X_test)
test_r2 = r2_score(y_test, test_pred)

print(f"Training set size: {len(X_train):,}, Test set size: {len(X_test):,}")
print(f"R-squared (train): {train_r2:.4f}")
print(f"R-squared (test):  {test_r2:.4f}")
print("(Similar train/test R^2 suggests the model isn't badly overfit; "
      "both being low means these features explain only a small share "
      "of the variance in popularity.)\n")

# ---------- Full-sample regression with statistical inference ----------
# Fit on the FULL cleaned dataset so coefficient estimates use all available
# data (standard practice: use the train/test split only to check
# generalization, then report inference from the full-sample fit).
X_full = np.column_stack([np.ones(len(X)), X])  # add intercept column
n, k = X_full.shape

# OLS via normal equations: beta = (X'X)^-1 X'y
XtX_inv = np.linalg.inv(X_full.T @ X_full)
beta = XtX_inv @ X_full.T @ y

y_pred_full = X_full @ beta
residuals = y - y_pred_full
rss = np.sum(residuals ** 2)
tss = np.sum((y - y.mean()) ** 2)
r_squared_full = 1 - rss / tss

# Standard errors, t-stats, p-values, 95% CIs for each coefficient
df_resid = n - k
sigma_squared = rss / df_resid
var_beta = sigma_squared * np.diag(XtX_inv)
se_beta = np.sqrt(var_beta)
t_stats = beta / se_beta
p_values = 2 * (1 - stats.t.cdf(np.abs(t_stats), df_resid))
ci_lower = beta - stats.t.ppf(0.975, df_resid) * se_beta
ci_upper = beta + stats.t.ppf(0.975, df_resid) * se_beta

coef_names = ["Intercept"] + features
results = pd.DataFrame({
    "feature": coef_names,
    "coefficient": beta,
    "std_error": se_beta,
    "t_stat": t_stats,
    "p_value": p_values,
    "ci_lower_95": ci_lower,
    "ci_upper_95": ci_upper,
})

pd.set_option("display.float_format", lambda x: f"{x:.4f}")
print("--- Full regression results (all 89,583 cleaned tracks) ---")
print(results.to_string(index=False))
print(f"\nR-squared (full sample): {r_squared_full:.4f}")

results.to_csv("regression_results.csv", index=False)
print("\nSaved detailed results to regression_results.csv")

print("""
--- Interpretation notes ---
This analysis identifies statistical ASSOCIATIONS between audio features and
Spotify popularity scores. It does NOT establish that changing an audio
feature would CAUSE a track's popularity to increase -- popularity is driven
by many factors outside this dataset (marketing, playlist placement, artist
fame, release timing, etc.) that are not controlled for here.

The low R-squared value indicates these eight audio features, together,
explain only a small portion of the variation in track popularity -- most
of what determines a song's popularity is not captured by its acoustic
characteristics alone.
""")
