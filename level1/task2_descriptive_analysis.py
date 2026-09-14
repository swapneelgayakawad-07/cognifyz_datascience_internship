"""
Cognifyz Data Science Internship
Level 1 - Task 2: Descriptive Analysis
"""

import pandas as pd

# -----------------------------------------------------
# Load the CLEANED dataset produced by Task 1
# -----------------------------------------------------
df = pd.read_csv("cleaned_dataset.csv")

# -----------------------------------------------------
# 1. Basic statistical measures for numerical columns
# -----------------------------------------------------
print("=" * 60)
print("1. BASIC STATISTICAL MEASURES (numerical columns)")
print("=" * 60)

numerical_cols = ["Average Cost for two", "Price range", "Aggregate rating", "Votes"]

stats = df[numerical_cols].agg(["mean", "median", "std", "min", "max"]).round(2)
print(stats)
print()

# -----------------------------------------------------
# 2. Distribution of categorical variables
# -----------------------------------------------------
print("=" * 60)
print("2. DISTRIBUTION OF CATEGORICAL VARIABLES")
print("=" * 60)

print("-- Country Code: number of unique values --")
print(f"{df['Country Code'].nunique()} unique country codes")
print(df["Country Code"].value_counts().head(10))
print()

print("-- City: number of unique values --")
print(f"{df['City'].nunique()} unique cities")
print(df["City"].value_counts().head(10))
print()

print("-- Cuisines: number of unique combinations --")
print(f"{df['Cuisines'].nunique()} unique cuisine combinations")
print(df["Cuisines"].value_counts().head(10))
print()

# -----------------------------------------------------
# 3. Top cuisines (individual, not combinations)
# -----------------------------------------------------
print("=" * 60)
print("3. TOP CUISINES (individual cuisine types)")
print("=" * 60)

# 'Cuisines' has comma-separated values like "French, Japanese, Desserts"
# Split them out so we count each individual cuisine, not each combination
all_cuisines = df["Cuisines"].str.split(",").explode().str.strip()
top_cuisines = all_cuisines.value_counts().head(10)
print(top_cuisines)
print()

# -----------------------------------------------------
# 4. Cities with the highest number of restaurants
# -----------------------------------------------------
print("=" * 60)
print("4. TOP CITIES BY NUMBER OF RESTAURANTS")
print("=" * 60)
top_cities = df["City"].value_counts().head(10)
print(top_cities)
print()

print("Observation: A single city (New Delhi and the wider NCR area - Gurgaon,")
print("Noida, Faridabad, Ghaziabad) dominates the dataset, since Zomato's data")
print("collection was heavily India-centric at the time. North Indian and Chinese")
print("cuisines are the most common individual cuisine types offered.")
