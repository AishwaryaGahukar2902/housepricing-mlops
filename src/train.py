import os
import joblib
import mlflow
import mlflow.sklearn

from sklearn.metrics import r2_score

from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor

from data_preprocessing import (
    preprocess_data,
    load_processed_data
)
# Run preprocessing
preprocess_data()

# Load data
X_train, X_test, y_train, y_test = load_processed_data()

# Models dictionary
models = {
    "LinearRegression": LinearRegression(),
    "DecisionTree": DecisionTreeRegressor(),
    "RandomForest": RandomForestRegressor( n_estimators=30,
    max_depth=8,
    random_state=42),
    "GradientBoosting": GradientBoostingRegressor()
}

best_score = -999
best_model = None
best_model_name = ""

# Create models folder
os.makedirs("../models", exist_ok=True)

# Train all models
for model_name, model in models.items():

    with mlflow.start_run(run_name=model_name):

        # Train
        model.fit(X_train, y_train)

        # Predict
        predictions = model.predict(X_test)

        # Accuracy
        score = r2_score(y_test, predictions)

        print(f"{model_name} R2 Score: {score}")

        # Log parameters
        mlflow.log_param("model_name", model_name)

        # Log metric
        mlflow.log_metric("r2_score", score)

        # Log model
        mlflow.sklearn.log_model(model, model_name)

        # Save best model
        if score > best_score:
            best_score = score
            best_model = model
            best_model_name = model_name

# Save best model locally
joblib.dump(best_model, "../models/model.pkl")

print("\nBest Model:", best_model_name)
print("Best Score:", best_score)

print("Best model saved successfully")