"""
Cognifyz Data Science Internship
Level 3 - Task 2: Customer Preference Analysis
"""

import pandas as pd

# -----------------------------------------------------
# Load the CLEANED dataset produced in Level 1 - Task 1
# -----------------------------------------------------
df = pd.read_csv("cleaned_dataset.csv")

# 'Cuisines' has comma-separated values like "North Indian, Chinese".
# To analyze per-cuisine, we split each restaurant's cuisines into separate
# rows (one row per restaurant-cuisine pair) so a restaurant with 3 cuisines
# contributes to the stats for all 3, not just lumped as one combination.
cuisine_df = df.assign(Cuisines=df["Cuisines"].str.split(",")).explode("Cuisines")
cuisine_df["Cuisines"] = cuisine_df["Cuisines"].str.strip()

print(f"Original rows: {len(df)}  ->  After splitting cuisines: {len(cuisine_df)}")
print()

# -----------------------------------------------------
# 1. Relationship between cuisine type and rating
# -----------------------------------------------------
print("=" * 60)
print("1. AVERAGE RATING BY CUISINE (min. 20 restaurants)")
print("=" * 60)

cuisine_counts = cuisine_df["Cuisines"].value_counts()
valid_cuisines = cuisine_counts[cuisine_counts >= 20].index

avg_rating_by_cuisine = (
    cuisine_df[cuisine_df["Cuisines"].isin(valid_cuisines)]
    .groupby("Cuisines")["Aggregate rating"]
    .mean()
    .sort_values(ascending=False)
)

print("Top 10 highest-rated cuisines:")
print(avg_rating_by_cuisine.head(10).round(2))
print()
print("Bottom 10 lowest-rated cuisines:")
print(avg_rating_by_cuisine.tail(10).round(2))
print()

# -----------------------------------------------------
# 2. Most popular cuisines by number of votes
# -----------------------------------------------------
print("=" * 60)
print("2. MOST POPULAR CUISINES BY TOTAL VOTES")
print("=" * 60)

votes_by_cuisine = (
    cuisine_df.groupby("Cuisines")["Votes"]
    .sum()
    .sort_values(ascending=False)
)
print(votes_by_cuisine.head(10))
print()
print("Note: this differs from 'most restaurants' (Level 1, Task 2) - this measures")
print("total customer ENGAGEMENT (votes), not just how many restaurants offer it.")
print()

# -----------------------------------------------------
# 3. Do specific cuisines get consistently higher ratings?
# -----------------------------------------------------
print("=" * 60)
print("3. CUISINES WITH HIGHEST RATING CONSISTENCY")
print("=" * 60)

cuisine_stats = (
    cuisine_df[cuisine_df["Cuisines"].isin(valid_cuisines)]
    .groupby("Cuisines")["Aggregate rating"]
    .agg(["mean", "std", "count"])
    .round(2)
    .sort_values("mean", ascending=False)
)
print("Top 10 cuisines by mean rating, with std (lower std = more consistent):")
print(cuisine_stats.head(10))
print()

top_cuisine_name = avg_rating_by_cuisine.index[0]
top_cuisine_count = cuisine_counts[top_cuisine_name]
print(f"Observation: '{top_cuisine_name}' tops the average-rating list, but with only")
print(f"{top_cuisine_count} restaurants meeting our 20+ threshold, so a handful of excellent")
print("spots can pull the average up more easily than for a huge category. Everyday")
print("cuisines like 'North Indian' and 'Fast Food' have thousands of restaurants and")
print("therefore a more reliable, but lower, average - reminding us that averages from")
print("small sample sizes need to be read with some caution.")
