"""
Cognifyz Data Science Internship
Level 1 - Task 1: Data Exploration and Preprocessing
"""

import pandas as pd

# -----------------------------------------------------
# 1. Loading the dataset
# -----------------------------------------------------
df = pd.read_csv("Dataset.csv", encoding="utf-8-sig")  # utf-8-sig strips the BOM in the first column name

print("=" * 60)
print("1. SHAPE OF THE DATASET")
print("=" * 60)
print(f"Rows: {df.shape[0]}")
print(f"Columns: {df.shape[1]}")
print()

# -----------------------------------------------------
# 2. Check missing values in each column
# -----------------------------------------------------
print("=" * 60)
print("2. MISSING VALUES PER COLUMN")
print("=" * 60)
missing = df.isnull().sum()
missing = missing[missing > 0]
if missing.empty:
    print("No missing values found in any column.")
else:
    print(missing)
print()

# Handle missing values
# 'Cuisines' is the only column with nulls in this dataset (9 rows).
# Since it's a categorical/text column and only a tiny fraction is missing,
# we fill it with "Unknown" rather than dropping rows (to avoid losing data).
if "Cuisines" in df.columns:
    df["Cuisines"] = df["Cuisines"].fillna("Unknown")

print("Missing values after handling:")
print(df.isnull().sum().sum(), "total missing values remain")
print()

# -----------------------------------------------------
# 3. Data type conversion
# -----------------------------------------------------
print("=" * 60)
print("3. DATA TYPES (before)")
print("=" * 60)
print(df.dtypes)
print()

# Convert columns that are logically categorical/boolean but stored as text
yes_no_cols = ["Has Table booking", "Has Online delivery", "Is delivering now", "Switch to order menu"]
for col in yes_no_cols:
    if col in df.columns:
        df[col] = df[col].map({"Yes": True, "No": False})

# Ensure numeric columns are properly typed
numeric_cols = ["Longitude", "Latitude", "Average Cost for two", "Price range", "Aggregate rating", "Votes"]
for col in numeric_cols:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

# Country Code / Restaurant ID are IDs, not measurements -> keep as int, but they act like categories
df["Country Code"] = df["Country Code"].astype("category")

print("DATA TYPES (after conversion)")
print(df.dtypes)
print()

# -----------------------------------------------------
# 4. Distribution of the target variable: Aggregate rating
# -----------------------------------------------------
print("=" * 60)
print("4. DISTRIBUTION OF 'Aggregate rating' (target variable)")
print("=" * 60)
print(df["Aggregate rating"].describe())
print()

# Bucket into rating bands to inspect class balance
bins = [-0.1, 0, 1, 2, 3, 3.5, 4, 4.5, 5]
labels = ["0 (Not rated)", "0-1", "1-2", "2-3", "3-3.5", "3.5-4", "4-4.5", "4.5-5"]
df["rating_band"] = pd.cut(df["Aggregate rating"], bins=bins, labels=labels)

band_counts = df["rating_band"].value_counts().sort_index()
band_pct = (band_counts / len(df) * 100).round(2)

print("Rating band counts and percentage of total:")
for band in labels:
    print(f"  {band:15s}: {band_counts.get(band, 0):5d} rows  ({band_pct.get(band, 0):5.2f}%)")
print()

zero_rating_pct = (df["Aggregate rating"] == 0).mean() * 100
print(f"Restaurants with a rating of exactly 0 (i.e. 'Not rated'): {zero_rating_pct:.2f}% of the dataset")
print()
print("Observation: There is a strong class imbalance - a large chunk of restaurants")
print("have an 'Aggregate rating' of 0 (not yet rated), while the rated restaurants")
print("cluster mostly between 3.0 and 4.0. Very few restaurants have ratings below 2")
print("(other than the 0 bucket) or above 4.5.")

# Save cleaned dataset for use in later tasks (Task 2, Task 3)
df.to_csv("cleaned_dataset.csv", index=False)
print()
print("Cleaned dataset saved to cleaned_dataset.csv")
