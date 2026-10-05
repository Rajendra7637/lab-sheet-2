

# %% [markdown]
# ## Setup: imports, paths and helper functions

# %%
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.preprocessing import (
    LabelEncoder,
    MaxAbsScaler,
    MinMaxScaler,
    RobustScaler,
    StandardScaler,
)

try:
    BASE_DIR = Path(__file__).resolve().parent
except NameError:
    BASE_DIR = Path.cwd()
    if BASE_DIR.name == "notebooks":
        BASE_DIR = BASE_DIR.parent

DATA_PATH = BASE_DIR / "datasets" / "housing_raw.csv"
PROCESSED_DIR = BASE_DIR / "processed"
OUT_DIR = BASE_DIR / "outputs"
PROCESSED_DIR.mkdir(exist_ok=True)
OUT_DIR.mkdir(exist_ok=True)

TARGET = "price_lakh"
NUMERIC_COLS = ["area_sqft", "bedrooms", "bathrooms", "age_years",
                "distance_to_metro_km", "price_lakh"]
CATEGORICAL_COLS = ["city", "property_type", "furnishing"]
OUTLIER_COLS = ["area_sqft", "distance_to_metro_km", "price_lakh"]


def load_dataset(path=DATA_PATH):
    """Load the CSV file with basic validation and exception handling."""
    try:
        if not Path(path).exists():
            raise FileNotFoundError(f"Dataset not found: {path}")
        data = pd.read_csv(path)
        if data.empty:
            raise ValueError("Dataset is empty")
        return data
    except (FileNotFoundError, ValueError, pd.errors.ParserError) as err:
        print("Error while loading dataset:", err)
        raise


def save_step(data, file_name):
    """Save the processed dataset after a major preprocessing step."""
    try:
        data.to_csv(PROCESSED_DIR / file_name, index=False)
        print(f"Saved -> processed/{file_name}  shape={data.shape}")
    except OSError as err:
        print("Could not save file:", err)


def iqr_bounds(series, factor=1.5):
    """Return lower and upper limits using the IQR method."""
    q1, q3 = series.quantile(0.25), series.quantile(0.75)
    iqr = q3 - q1
    return q1 - factor * iqr, q3 + factor * iqr


# %% [markdown]
# # Part A: Handling Missing Values (Programs 1-10)

# %% [markdown]
# ## Program 1: Load a dataset and identify missing values in each column

# %%
original_df = load_dataset()
print("Shape:", original_df.shape)
print("\nMissing values in each column:")
print(original_df.isnull().sum())

# %% [markdown]
# ## Program 2: Percentage of missing values in every feature

# %%
missing_percent = (original_df.isnull().mean() * 100).round(2)
print(missing_percent.sort_values(ascending=False))

# %% [markdown]
# ## Program 3: Remove rows containing missing values

# %%
rows_removed_df = original_df.copy().dropna()
print("Rows before:", len(original_df), "| Rows after:", len(rows_removed_df))
print("Rows lost  :", len(original_df) - len(rows_removed_df))

# %% [markdown]
# ## Program 4: Remove columns having more than 50% missing values

# %%
columns_dropped_df = original_df.copy()
missing_ratio = columns_dropped_df.isnull().mean()
columns_to_drop = missing_ratio[missing_ratio > 0.5].index.tolist()
columns_dropped_df = columns_dropped_df.drop(columns=columns_to_drop)
print("Columns dropped:", columns_to_drop)
print("Shape before:", original_df.shape, "| after:", columns_dropped_df.shape)

# %% [markdown]
# ## Program 5: Replace missing numerical values using the mean

# %%
mean_filled_df = columns_dropped_df.copy()
for col in NUMERIC_COLS:
    mean_filled_df[col] = mean_filled_df[col].fillna(mean_filled_df[col].mean())
print(mean_filled_df[NUMERIC_COLS].isnull().sum())

# %% [markdown]
# ## Program 6: Replace missing numerical values using the median

# %%
median_filled_df = columns_dropped_df.copy()
for col in NUMERIC_COLS:
    median_filled_df[col] = median_filled_df[col].fillna(median_filled_df[col].median())
print(median_filled_df[NUMERIC_COLS].isnull().sum())

# %% [markdown]
# ## Program 7: Replace missing categorical values using the mode

# %%
mode_filled_df = columns_dropped_df.copy()
for col in CATEGORICAL_COLS:
    mode_value = mode_filled_df[col].mode()[0]
    mode_filled_df[col] = mode_filled_df[col].fillna(mode_value)
    print(f"{col}: filled with mode = '{mode_value}'")
print(mode_filled_df[CATEGORICAL_COLS].isnull().sum())

# %% [markdown]
# ## Program 8: Fill missing values using forward fill

# %%
ffill_df = columns_dropped_df.copy().ffill()
print("Missing values after forward fill:")
print(ffill_df.isnull().sum()[lambda s: s > 0])  # only the first row can stay empty

# %% [markdown]
# ## Program 9: Fill missing values using backward fill

# %%
bfill_df = columns_dropped_df.copy().bfill()
print("Missing values after backward fill:")
print(bfill_df.isnull().sum()[lambda s: s > 0])  # only the last row can stay empty

# %% [markdown]
# ## Program 10: Compare the dataset before and after handling missing values

# %%
# Final choice: median for numbers, mode for categories
clean_df = columns_dropped_df.copy()
for col in NUMERIC_COLS:
    clean_df[col] = clean_df[col].fillna(clean_df[col].median())
for col in CATEGORICAL_COLS:
    clean_df[col] = clean_df[col].fillna(clean_df[col].mode()[0])
for col in ["bedrooms", "bathrooms", "age_years"]:
    clean_df[col] = clean_df[col].round().astype(int)

comparison = pd.DataFrame({
    "missing_before": original_df.isnull().sum(),
    "missing_after": clean_df.reindex(columns=original_df.columns).isnull().sum(),
})
print(comparison)
print("\nMean before:\n", original_df[NUMERIC_COLS].mean().round(2))
print("\nMean after:\n", clean_df[NUMERIC_COLS].mean().round(2))
save_step(clean_df, "01_missing_handled.csv")

# %% [markdown]
# # Part B: Outliers (Programs 11-18)

# %% [markdown]
# ## Program 11: Detect outliers using the IQR method

# %%
for col in OUTLIER_COLS:
    low, high = iqr_bounds(clean_df[col])
    outliers = clean_df[(clean_df[col] < low) | (clean_df[col] > high)]
    print(f"{col}: limits=({low:.1f}, {high:.1f}) -> {len(outliers)} outliers")

# %% [markdown]
# ## Program 12: Detect outliers using the Z-score method

# %%
for col in OUTLIER_COLS:
    z_scores = (clean_df[col] - clean_df[col].mean()) / clean_df[col].std()
    print(f"{col}: {(z_scores.abs() > 3).sum()} outliers (|z| > 3)")

# %% [markdown]
# ## Program 13: Visualize outliers using a Box Plot

# %%
fig, axes = plt.subplots(1, 3, figsize=(12, 4))
for ax, col in zip(axes, OUTLIER_COLS):
    sns.boxplot(y=clean_df[col], ax=ax, color="lightblue")
    ax.set_title(f"Box plot: {col}")
plt.tight_layout()
plt.savefig(OUT_DIR / "p13_boxplot_outliers.png", dpi=120)
plt.show()

# %% [markdown]
# ## Program 14: Visualize outliers using a Scatter Plot

# %%
low, high = iqr_bounds(clean_df["price_lakh"])
is_outlier = (clean_df["price_lakh"] < low) | (clean_df["price_lakh"] > high)
plt.figure(figsize=(7, 5))
plt.scatter(clean_df.loc[~is_outlier, "area_sqft"], clean_df.loc[~is_outlier, "price_lakh"],
            alpha=0.6, label="Normal")
plt.scatter(clean_df.loc[is_outlier, "area_sqft"], clean_df.loc[is_outlier, "price_lakh"],
            color="red", label="Outlier")
plt.xlabel("Area (sqft)")
plt.ylabel("Price (lakh)")
plt.title("Scatter Plot: Area vs Price")
plt.legend()
plt.savefig(OUT_DIR / "p14_scatter_outliers.png", dpi=120)
plt.show()

# %% [markdown]
# ## Program 15: Remove outliers using the IQR method

# %%
outliers_removed_df = clean_df.copy()
keep_mask = pd.Series(True, index=outliers_removed_df.index)
for col in OUTLIER_COLS:
    low, high = iqr_bounds(outliers_removed_df[col])
    keep_mask &= outliers_removed_df[col].between(low, high)
outliers_removed_df = outliers_removed_df[keep_mask]
print("Rows before:", len(clean_df), "| Rows after:", len(outliers_removed_df))

# %% [markdown]
# ## Program 16: Replace outliers with the median value

# %%
median_replaced_df = clean_df.copy()
for col in OUTLIER_COLS:
    low, high = iqr_bounds(median_replaced_df[col])
    outside = (median_replaced_df[col] < low) | (median_replaced_df[col] > high)
    median_replaced_df.loc[outside, col] = median_replaced_df[col].median()
    print(f"{col}: {outside.sum()} values replaced with median")

# %% [markdown]
# ## Program 17: Cap outliers using percentile-based capping

# %%
capped_df = clean_df.copy()
for col in OUTLIER_COLS:
    lower_cap = capped_df[col].quantile(0.01)
    upper_cap = capped_df[col].quantile(0.99)
    capped_df[col] = capped_df[col].clip(lower=lower_cap, upper=upper_cap)
    print(f"{col}: capped between {lower_cap:.1f} and {upper_cap:.1f}")

# %% [markdown]
# ## Program 18: Compare the dataset before and after outlier treatment

# %%
summary = pd.DataFrame({
    "original": clean_df["price_lakh"].describe(),
    "removed": outliers_removed_df["price_lakh"].describe(),
    "median_replaced": median_replaced_df["price_lakh"].describe(),
    "capped": capped_df["price_lakh"].describe(),
}).round(2)
print(summary)

plt.figure(figsize=(8, 4))
sns.boxplot(data=[clean_df["price_lakh"], outliers_removed_df["price_lakh"],
                  median_replaced_df["price_lakh"], capped_df["price_lakh"]])
plt.xticks([0, 1, 2, 3], ["Original", "Removed", "Median", "Capped"])
plt.title("Price before and after outlier treatment")
plt.savefig(OUT_DIR / "p18_outlier_comparison.png", dpi=120)
plt.show()

# Use the capped data (no rows lost) for the next parts
treated_df = capped_df.copy()
save_step(treated_df, "02_outliers_treated.csv")

# %% [markdown]
# # Part C: Normalization and Scaling (Programs 19-26)

# %% [markdown]
# ## Program 19: Min-Max Normalization

# %%
minmax_df = treated_df.copy()
minmax_df[NUMERIC_COLS] = MinMaxScaler().fit_transform(minmax_df[NUMERIC_COLS])
print(minmax_df[NUMERIC_COLS].describe().loc[["min", "max"]])

# %% [markdown]
# ## Program 20: Standardization (Z-score Scaling)

# %%
standard_df = treated_df.copy()
standard_df[NUMERIC_COLS] = StandardScaler().fit_transform(standard_df[NUMERIC_COLS])
print(standard_df[NUMERIC_COLS].describe().loc[["mean", "std"]].round(3))

# %% [markdown]
# ## Program 21: Robust Scaling to handle outliers

# %%
robust_df = treated_df.copy()
robust_df[NUMERIC_COLS] = RobustScaler().fit_transform(robust_df[NUMERIC_COLS])
print(robust_df[NUMERIC_COLS].describe().loc[["25%", "50%", "75%"]].round(3))

# %% [markdown]
# ## Program 22: Max Absolute Scaling

# %%
maxabs_df = treated_df.copy()
maxabs_df[NUMERIC_COLS] = MaxAbsScaler().fit_transform(maxabs_df[NUMERIC_COLS])
print(maxabs_df[NUMERIC_COLS].abs().max())

# %% [markdown]
# ## Program 23: Compare original and normalized datasets

# %%
print("ORIGINAL:")
print(treated_df[NUMERIC_COLS].describe().loc[["min", "max", "mean", "std"]].round(2))
print("\nMIN-MAX NORMALIZED:")
print(minmax_df[NUMERIC_COLS].describe().loc[["min", "max", "mean", "std"]].round(2))

# %% [markdown]
# ## Program 24: Effect of normalization using histograms

# %%
fig, axes = plt.subplots(1, 2, figsize=(11, 4))
axes[0].hist(treated_df["area_sqft"], bins=20, color="salmon", edgecolor="black")
axes[0].set_title("area_sqft - Original")
axes[1].hist(minmax_df["area_sqft"], bins=20, color="lightgreen", edgecolor="black")
axes[1].set_title("area_sqft - Min-Max Normalized")
plt.tight_layout()
plt.savefig(OUT_DIR / "p24_normalization_histograms.png", dpi=120)
plt.show()

# %% [markdown]
# ## Program 25: Effect of scaling using box plots

# %%
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
sns.boxplot(data=treated_df[NUMERIC_COLS], ax=axes[0])
axes[0].set_title("Original (different ranges)")
axes[0].tick_params(axis="x", rotation=45)
sns.boxplot(data=standard_df[NUMERIC_COLS], ax=axes[1])
axes[1].set_title("After Standardization (same scale)")
axes[1].tick_params(axis="x", rotation=45)
plt.tight_layout()
plt.savefig(OUT_DIR / "p25_scaling_boxplots.png", dpi=120)
plt.show()

# %% [markdown]
# ## Program 26: Compare different scaling techniques on the same dataset

# %%
scaled_versions = {
    "Original": treated_df,
    "MinMax": minmax_df,
    "Standard": standard_df,
    "Robust": robust_df,
    "MaxAbs": maxabs_df,
}
compare_table = pd.DataFrame({
    name: data["area_sqft"].agg(["min", "max", "mean", "std"])
    for name, data in scaled_versions.items()
}).round(3).T
print(compare_table)

fig, axes = plt.subplots(1, 5, figsize=(18, 3))
for ax, (name, data) in zip(axes, scaled_versions.items()):
    sns.histplot(data["area_sqft"], bins=20, ax=ax, color="steelblue")
    ax.set_title(name)
    ax.set_xlabel("")
plt.tight_layout()
plt.savefig(OUT_DIR / "p26_scaling_comparison.png", dpi=120)
plt.show()
save_step(standard_df, "03_standardized.csv")

# %% [markdown]
# # Part D: Encoding and Feature Engineering (Programs 27-35)

# %% [markdown]
# ## Program 27: Label Encoding

# %%
label_df = treated_df.copy()
label_encoder = LabelEncoder()
label_df["city_label"] = label_encoder.fit_transform(label_df["city"])
print(dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_))))
print(label_df[["city", "city_label"]].head())

# %% [markdown]
# ## Program 28: One-Hot Encoding

# %%
onehot_df = pd.get_dummies(treated_df, columns=["property_type"], dtype=int)
print([c for c in onehot_df.columns if c.startswith("property_type_")])
print(onehot_df.filter(like="property_type_").head())

# %% [markdown]
# ## Program 29: Binary Encoding

# %%
def binary_encode(series, prefix):
    """Convert each category to a number, then write the number in binary bits."""
    codes, categories = pd.factorize(series)
    bit_count = max(1, int(np.ceil(np.log2(len(categories)))))
    bits = (codes[:, None] >> np.arange(bit_count - 1, -1, -1)) & 1
    columns = [f"{prefix}_bit{i + 1}" for i in range(bit_count)]
    return pd.DataFrame(bits, columns=columns, index=series.index), categories


binary_bits, city_categories = binary_encode(treated_df["city"], "city")
binary_df = pd.concat([treated_df[["city"]], binary_bits], axis=1)
print("Categories:", list(city_categories))
print(binary_df.drop_duplicates().sort_values("city"))

# %% [markdown]
# ## Program 30: Create a new feature by combining two columns

# %%
feature_df = treated_df.copy()
feature_df["total_rooms"] = feature_df["bedrooms"] + feature_df["bathrooms"]
print(feature_df[["bedrooms", "bathrooms", "total_rooms"]].head())

# %% [markdown]
# ## Program 31: Extract year, month and day from a date column

# %%
feature_df["listing_date"] = pd.to_datetime(feature_df["listing_date"])
feature_df["listing_year"] = feature_df["listing_date"].dt.year
feature_df["listing_month"] = feature_df["listing_date"].dt.month
feature_df["listing_day"] = feature_df["listing_date"].dt.day
print(feature_df[["listing_date", "listing_year", "listing_month", "listing_day"]].head())

# %% [markdown]
# ## Program 32: Create a new feature using mathematical transformations

# %%
feature_df["price_per_sqft"] = (feature_df["price_lakh"] * 100000 / feature_df["area_sqft"]).round(2)
feature_df["area_sqrt"] = np.sqrt(feature_df["area_sqft"]).round(2)
print(feature_df[["area_sqft", "area_sqrt", "price_lakh", "price_per_sqft"]].head())

# %% [markdown]
# ## Program 33: Log Transformation on skewed data

# %%
print("Skewness before log:")
print(feature_df[["area_sqft", "distance_to_metro_km"]].skew().round(3))

feature_df["area_log"] = np.log1p(feature_df["area_sqft"])
feature_df["distance_log"] = np.log1p(feature_df["distance_to_metro_km"])
print("\nSkewness after log:")
print(feature_df[["area_log", "distance_log"]].skew().round(3))

fig, axes = plt.subplots(1, 2, figsize=(11, 4))
sns.histplot(feature_df["distance_to_metro_km"], bins=20, kde=True, ax=axes[0])
axes[0].set_title("distance - before log")
sns.histplot(feature_df["distance_log"], bins=20, kde=True, ax=axes[1])
axes[1].set_title("distance - after log")
plt.tight_layout()
plt.savefig(OUT_DIR / "p33_log_transform.png", dpi=120)
plt.show()

# %% [markdown]
# ## Program 34: Feature Selection using correlation analysis

# %%
candidate_df = feature_df.select_dtypes(include="number").drop(columns=["house_id"])
corr_matrix = candidate_df.corr()

# Step 1: correlation of every feature with the target
target_corr = corr_matrix[TARGET].drop(TARGET).abs().sort_values(ascending=False)
print("Correlation with target:\n", target_corr.round(3))

# Step 2: drop one feature from every highly correlated pair (> 0.9)
upper = corr_matrix.drop(columns=[TARGET], index=[TARGET]).abs()
upper = upper.where(np.triu(np.ones(upper.shape), k=1).astype(bool))
redundant = [col for col in upper.columns if (upper[col] > 0.9).any()]
print("\nRedundant features (corr > 0.9 with another feature):", redundant)

selected_features = [c for c in target_corr.index if c not in redundant]
print("Selected features:", selected_features)

plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm", annot_kws={"size": 7})
plt.title("Correlation Heatmap")
plt.savefig(OUT_DIR / "p34_correlation_heatmap.png", dpi=120, bbox_inches="tight")
plt.show()

# %% [markdown]
# ## Program 35: Final preprocessed dataset ready for Machine Learning

# %%
final_df = treated_df.copy()

# Feature engineering
final_df["total_rooms"] = final_df["bedrooms"] + final_df["bathrooms"]
dates = pd.to_datetime(final_df["listing_date"])
final_df["listing_year"] = dates.dt.year
final_df["listing_month"] = dates.dt.month
final_df["listing_day"] = dates.dt.day
final_df["area_log"] = np.log1p(final_df["area_sqft"])
final_df["distance_log"] = np.log1p(final_df["distance_to_metro_km"])

# Drop columns that are not useful for the model
final_df = final_df.drop(columns=["house_id", "listing_date", "area_sqft", "distance_to_metro_km"])

# Encode categorical columns
final_df = pd.get_dummies(final_df, columns=CATEGORICAL_COLS, drop_first=True, dtype=int)

# Scale the numerical features (target is kept in original units)
features_to_scale = ["bedrooms", "bathrooms", "age_years", "total_rooms",
                     "listing_year", "listing_month", "listing_day",
                     "area_log", "distance_log"]
final_df[features_to_scale] = StandardScaler().fit_transform(final_df[features_to_scale])

# Remove highly correlated features
corr_final = final_df.drop(columns=[TARGET]).corr().abs()
upper_final = corr_final.where(np.triu(np.ones(corr_final.shape), k=1).astype(bool))
to_drop = [c for c in upper_final.columns if (upper_final[c] > 0.9).any()]
final_df = final_df.drop(columns=to_drop)
print("Dropped for high correlation:", to_drop)

# Final checks
assert final_df.isnull().sum().sum() == 0, "Missing values still present"
print("\nFinal shape:", final_df.shape)
print(final_df.head())
save_step(final_df, "final_preprocessed.csv")

# %% [markdown]
# ## Conclusion
# All 35 experiments of Lab Sheet-02 were completed. Missing values and outliers
# were handled, features were scaled and encoded, new features were created,
# and a clean dataset ready for machine learning was saved.
