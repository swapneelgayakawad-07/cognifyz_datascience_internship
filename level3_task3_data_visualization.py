"""
Cognifyz Data Science Internship
Level 3 - Task 3: Data Visualization
"""

import pandas as pd
import matplotlib.pyplot as plt

# -----------------------------------------------------
# Load the CLEANED dataset produced in Level 1 - Task 1
# -----------------------------------------------------
df = pd.read_csv("cleaned_dataset.csv")

# -----------------------------------------------------
# 1. Distribution of ratings - histogram
# -----------------------------------------------------
plt.figure(figsize=(9, 5))
plt.hist(df["Aggregate rating"], bins=20, color="teal", edgecolor="black")
plt.xlabel("Aggregate rating")
plt.ylabel("Number of restaurants")
plt.title("Distribution of Aggregate Ratings (Histogram)")
plt.tight_layout()
plt.savefig("viz_rating_histogram.png", dpi=150)
plt.close()
print("Saved: viz_rating_histogram.png")

# Distribution of ratings - bar plot (excluding the 0/unrated spike, which we
# already know dominates, so we can see the shape of the RATED restaurants clearly)
rated_df = df[df["Aggregate rating"] > 0]
bins = [0, 1, 2, 3, 3.5, 4, 4.5, 5]
labels = ["0-1", "1-2", "2-3", "3-3.5", "3.5-4", "4-4.5", "4.5-5"]
rated_df = rated_df.copy()
rated_df["band"] = pd.cut(rated_df["Aggregate rating"], bins=bins, labels=labels)
band_counts = rated_df["band"].value_counts().reindex(labels)

plt.figure(figsize=(9, 5))
band_counts.plot(kind="bar", color="coral", edgecolor="black")
plt.xlabel("Rating band")
plt.ylabel("Number of restaurants")
plt.title("Distribution of Ratings Among RATED Restaurants (Bar Plot)")
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig("viz_rating_barplot.png", dpi=150)
plt.close()
print("Saved: viz_rating_barplot.png")
print()

# -----------------------------------------------------
# 2. Compare average ratings across cuisines and cities
# -----------------------------------------------------
cuisine_df = df.assign(Cuisines=df["Cuisines"].str.split(",")).explode("Cuisines")
cuisine_df["Cuisines"] = cuisine_df["Cuisines"].str.strip()
cuisine_counts = cuisine_df["Cuisines"].value_counts()
valid_cuisines = cuisine_counts[cuisine_counts >= 20].index

top10_cuisine_rating = (
    cuisine_df[cuisine_df["Cuisines"].isin(valid_cuisines)]
    .groupby("Cuisines")["Aggregate rating"]
    .mean()
    .sort_values(ascending=False)
    .head(10)
)

plt.figure(figsize=(9, 5))
top10_cuisine_rating.plot(kind="barh", color="mediumseagreen")
plt.xlabel("Average rating")
plt.title("Top 10 Cuisines by Average Rating (min. 20 restaurants)")
plt.gca().invert_yaxis()  # highest at top
plt.tight_layout()
plt.savefig("viz_top_cuisines_rating.png", dpi=150)
plt.close()
print("Saved: viz_top_cuisines_rating.png")

city_counts = df["City"].value_counts()
valid_cities = city_counts[city_counts >= 20].index
top10_city_rating = (
    df[df["City"].isin(valid_cities)]
    .groupby("City")["Aggregate rating"]
    .mean()
    .sort_values(ascending=False)
    .head(10)
)

plt.figure(figsize=(9, 5))
top10_city_rating.plot(kind="barh", color="cornflowerblue")
plt.xlabel("Average rating")
plt.title("Top 10 Cities by Average Rating (min. 20 restaurants)")
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig("viz_top_cities_rating.png", dpi=150)
plt.close()
print("Saved: viz_top_cities_rating.png")
print()

# -----------------------------------------------------
# 3. Relationship between features and the target variable
# -----------------------------------------------------
# Votes vs Rating - does a more "popular" (heavily-voted) restaurant tend to
# have a higher rating?
plt.figure(figsize=(9, 5))
plt.scatter(df["Votes"], df["Aggregate rating"], alpha=0.3, s=10, color="darkorange")
plt.xlabel("Votes")
plt.ylabel("Aggregate rating")
plt.title("Votes vs. Aggregate Rating")
plt.tight_layout()
plt.savefig("viz_votes_vs_rating.png", dpi=150)
plt.close()
print("Saved: viz_votes_vs_rating.png")

# Average Cost for two vs Rating (log scale on cost, since it has huge outliers)
plt.figure(figsize=(9, 5))
plt.scatter(df["Average Cost for two"], df["Aggregate rating"], alpha=0.3, s=10, color="purple")
plt.xscale("log")
plt.xlabel("Average Cost for two (log scale)")
plt.ylabel("Aggregate rating")
plt.title("Cost for Two vs. Aggregate Rating")
plt.tight_layout()
plt.savefig("viz_cost_vs_rating.png", dpi=150)
plt.close()
print("Saved: viz_cost_vs_rating.png")
print()

# Price range vs average rating (bar plot, since price range is a small set of categories 1-4)
price_avg = df.groupby("Price range")["Aggregate rating"].mean()
plt.figure(figsize=(7, 5))
price_avg.plot(kind="bar", color="slateblue", edgecolor="black")
plt.xlabel("Price range (1=cheapest, 4=most expensive)")
plt.ylabel("Average rating")
plt.title("Average Rating by Price Range")
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig("viz_price_range_vs_rating.png", dpi=150)
plt.close()
print("Saved: viz_price_range_vs_rating.png")
print()

print("Observation: There's a clear upward trend between Price range and average")
print("rating - pricier restaurants (range 4) tend to be rated notably higher than")
print("cheap ones (range 1). Votes also shows a mild positive relationship with")
print("rating, though there's a lot of scatter. Cost for two shows a similar mild")
print("upward trend once plotted on a log scale (needed because a few extreme")
print("outliers would otherwise squash the whole chart).")
