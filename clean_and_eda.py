"""
Part 1a: Data Cleaning & Exploratory Analysis
Spotify Recommendation & Engagement Analysis

Dataset: Kaggle "Spotify Tracks Dataset" (114,000 tracks)
Source: https://www.kaggle.com/datasets/maharshipandya/-spotify-tracks-dataset
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style("whitegrid")

# ---------- LOAD ----------
df = pd.read_csv("dataset.csv")
print(f"Raw dataset shape: {df.shape}")

# ---------- CLEANING ----------
# 1. Drop the unnamed index column (artifact of how the CSV was exported)
df = df.drop(columns=["Unnamed: 0"])

# 2. Drop the single row with missing track/artist/album info
before = len(df)
df = df.dropna()
print(f"Dropped {before - len(df)} row(s) with missing values")

# 3. De-duplicate tracks. This dataset lists the same track multiple times
#    under different genre tags (e.g. a song tagged both "rock" and "alt-rock").
#    Since our analysis is about audio features and popularity, not genre,
#    we keep only the first occurrence of each track_id.
before = len(df)
df = df.drop_duplicates(subset="track_id", keep="first")
print(f"Dropped {before - len(df)} duplicate track_id rows (same song under multiple genres)")

# 4. Remove tracks with tempo == 0. A tempo of exactly 0 BPM is not a real
#    musical value -- it indicates Spotify's audio analysis failed to detect
#    a tempo for that track, not that the track has no tempo.
before = len(df)
df = df[df["tempo"] > 0]
print(f"Dropped {before - len(df)} rows with tempo == 0 (likely failed audio analysis)")

print(f"\nFinal cleaned dataset shape: {df.shape}")

# Note on popularity == 0 tracks: we keep these. A popularity score of 0
# is a plausible, real value (an obscure or rarely-played track), not
# necessarily an error, so removing them would bias the dataset toward
# only well-known songs.
zero_pop_count = (df["popularity"] == 0).sum()
print(f"Tracks with popularity == 0 (kept, not an error): {zero_pop_count}")

df.to_csv("dataset_cleaned.csv", index=False)
print("\nSaved cleaned dataset to dataset_cleaned.csv")

# ---------- EXPLORATORY ANALYSIS ----------
features = ["danceability", "energy", "valence", "tempo", "acousticness",
            "loudness", "instrumentalness", "speechiness"]

fig, axes = plt.subplots(2, 4, figsize=(20, 9))
axes = axes.flatten()
for i, feat in enumerate(features):
    axes[i].hist(df[feat], bins=40, color="#1DB954", edgecolor="black", alpha=0.8)
    axes[i].set_title(f"Distribution of {feat}")
plt.tight_layout()
plt.savefig("eda_feature_distributions.png", dpi=150)
print("Saved eda_feature_distributions.png")

# Correlation heatmap: popularity vs. audio features
corr_cols = ["popularity"] + features
corr_matrix = df[corr_cols].corr()

plt.figure(figsize=(9, 7))
sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="RdYlGn", center=0,
            square=True, linewidths=0.5)
plt.title("Correlation: Popularity vs. Audio Features")
plt.tight_layout()
plt.savefig("eda_correlation_heatmap.png", dpi=150)
print("Saved eda_correlation_heatmap.png")

print("\n--- Simple correlations with popularity (for reference only -- see regression for a fuller picture) ---")
print(corr_matrix["popularity"].sort_values(ascending=False))
