"""
Cognifyz Data Science Internship
Level 3 - Task 1: Predictive Modeling
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# -----------------------------------------------------
# Load the CLEANED dataset produced in Level 1 - Task 1
# -----------------------------------------------------
df = pd.read_csv("cleaned_dataset.csv")

# -----------------------------------------------------
# 1. Select features (X) and target (y)
# -----------------------------------------------------
# We predict 'Aggregate rating' using numeric/boolean columns that make sense
# as inputs. We exclude columns that are IDs, free text, or that leak the
# answer (like 'Rating color'/'Rating text', which are DERIVED FROM the rating
# itself - including them would be cheating, not predicting).
feature_cols = [
    "Average Cost for two",
    "Price range",
    "Votes",
    "Has Table booking",
    "Has Online delivery",
]
target_col = "Aggregate rating"

X = df[feature_cols].copy()
y = df[target_col].copy()

# Convert booleans to 0/1 so the models can use them numerically
X["Has Table booking"] = X["Has Table booking"].astype(int)
X["Has Online delivery"] = X["Has Online delivery"].astype(int)

print("Features used:", feature_cols)
print("Target:", target_col)
print()

# -----------------------------------------------------
# 2. Train/test split
# -----------------------------------------------------
# We hold out 20% of the data to test on - the model never sees this during
# training, so testing on it tells us how well it generalizes to new data.
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"Training rows: {len(X_train)}")
print(f"Testing rows:  {len(X_test)}")
print()

# -----------------------------------------------------
# 3. Train and evaluate multiple models
# -----------------------------------------------------
models = {
    "Linear Regression": LinearRegression(),
    "Decision Tree": DecisionTreeRegressor(random_state=42, max_depth=6),
    "Random Forest": RandomForestRegressor(random_state=42, n_estimators=200, max_depth=8),
}

print("=" * 60)
print("MODEL COMPARISON")
print("=" * 60)

results = []
for name, model in models.items():
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    mse = mean_squared_error(y_test, predictions)
    rmse = mse ** 0.5
    r2 = r2_score(y_test, predictions)

    results.append({"Model": name, "RMSE": round(rmse, 3), "R2 Score": round(r2, 3)})
    print(f"\n{name}")
    print(f"  RMSE (Root Mean Squared Error): {rmse:.3f}  <- lower is better, same units as rating (0-5)")
    print(f"  R^2 Score:                      {r2:.3f}  <- closer to 1.0 is better, 0 means no better than guessing the average")

print()
print("=" * 60)
print("SUMMARY TABLE")
print("=" * 60)
results_df = pd.DataFrame(results).sort_values("R2 Score", ascending=False)
print(results_df.to_string(index=False))
print()

best_model_name = results_df.iloc[0]["Model"]
print(f"Best performing model: {best_model_name}")
print()
print("Observation: Random Forest / Decision Tree usually outperform plain Linear")
print("Regression here because the relationship between votes/cost/price-range and")
print("rating isn't purely linear - tree-based models can capture more complex,")
print("non-linear patterns in the data.")
