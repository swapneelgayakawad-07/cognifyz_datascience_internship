"""
Cognifyz Data Science Internship
Level 1 - Task 3: Geospatial Analysis
"""

import pandas as pd
import matplotlib.pyplot as plt

# -----------------------------------------------------
# Load the CLEANED dataset produced by Task 1
# -----------------------------------------------------
df = pd.read_csv("cleaned_dataset.csv")

# Drop rows with missing/invalid coordinates (0,0 is not a real restaurant location)
geo_df = df[(df["Latitude"] != 0) & (df["Longitude"] != 0)].copy()
print(f"Using {len(geo_df)} of {len(df)} rows with valid coordinates.\n")

# -----------------------------------------------------
# 1. Visualize restaurant locations on a map (scatter plot as a simple map)
# -----------------------------------------------------
print("=" * 60)
print("1. PLOTTING RESTAURANT LOCATIONS")
print("=" * 60)

plt.figure(figsize=(10, 6))
scatter = plt.scatter(
    geo_df["Longitude"], geo_df["Latitude"],
    c=geo_df["Aggregate rating"], cmap="RdYlGn",
    s=8, alpha=0.6
)
plt.colorbar(scatter, label="Aggregate rating")
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.title("Restaurant Locations Worldwide (colored by rating)")
plt.tight_layout()
plt.savefig("restaurant_locations_world.png", dpi=150)
plt.close()
print("Saved: restaurant_locations_world.png (world view - shows India cluster dominates)")

# Zoomed-in view of India/Delhi-NCR, since that's ~90% of the data
india_df = geo_df[(geo_df["Longitude"].between(68, 98)) & (geo_df["Latitude"].between(6, 38))]
plt.figure(figsize=(10, 6))
scatter = plt.scatter(
    india_df["Longitude"], india_df["Latitude"],
    c=india_df["Aggregate rating"], cmap="RdYlGn",
    s=8, alpha=0.6
)
plt.colorbar(scatter, label="Aggregate rating")
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.title("Restaurant Locations - India (colored by rating)")
plt.tight_layout()
plt.savefig("restaurant_locations_india.png", dpi=150)
plt.close()
print("Saved: restaurant_locations_india.png (zoomed to India, where most data is)")
print()

# -----------------------------------------------------
# 2. Distribution of restaurants across cities/countries
# -----------------------------------------------------
print("=" * 60)
print("2. DISTRIBUTION ACROSS CITIES AND COUNTRIES")
print("=" * 60)

top_cities = df["City"].value_counts().head(10)
print("Top 10 cities by restaurant count:")
print(top_cities)
print()

plt.figure(figsize=(10, 6))
top_cities.plot(kind="bar", color="steelblue")
plt.ylabel("Number of restaurants")
plt.title("Top 10 Cities by Restaurant Count")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.savefig("top_cities_barchart.png", dpi=150)
plt.close()
print("Saved: top_cities_barchart.png")
print()

country_counts = df["Country Code"].value_counts()
print(f"Restaurants are spread across {df['Country Code'].nunique()} countries, but")
print(f"country code 1 (India) alone accounts for {country_counts.iloc[0]} of {len(df)} rows")
print(f"({country_counts.iloc[0] / len(df) * 100:.1f}%).")
print()

# -----------------------------------------------------
# 3. Correlation between restaurant location and rating
# -----------------------------------------------------
print("=" * 60)
print("3. CORRELATION BETWEEN LOCATION AND RATING")
print("=" * 60)

corr_lat = geo_df["Latitude"].corr(geo_df["Aggregate rating"])
corr_lon = geo_df["Longitude"].corr(geo_df["Aggregate rating"])

print(f"Correlation (Latitude  vs Aggregate rating): {corr_lat:.4f}")
print(f"Correlation (Longitude vs Aggregate rating): {corr_lon:.4f}")
print()
print("Observation: Both correlations are close to 0, meaning there is no strong")
print("LINEAR relationship between a restaurant's raw coordinates and its rating -")
print("which makes sense, since latitude/longitude are just position labels, not")
print("meaningful numeric quantities. A more useful analysis is comparing rating")
print("BY CITY (grouped average) rather than by raw coordinate value:")
print()

city_avg_rating = df.groupby("City")["Aggregate rating"].mean().sort_values(ascending=False)
print("Top 10 cities by AVERAGE rating (min. 5 restaurants):")
city_counts = df["City"].value_counts()
valid_cities = city_counts[city_counts >= 5].index
print(city_avg_rating[city_avg_rating.index.isin(valid_cities)].head(10).round(2))
