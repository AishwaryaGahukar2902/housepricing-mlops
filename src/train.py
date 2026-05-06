import os
import joblib
import mlflow
import mlflow.sklearn

from sklearn.metrics import (
    r2_score,
    mean_absolute_error,
    mean_squared_error
)

from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import (
    RandomForestRegressor,
    GradientBoostingRegressor
)

from xgboost import XGBRegressor

from data_preprocessing import (
    preprocess_data,
    load_processed_data
)

# =========================================================
# Set MLflow Experiment
# =========================================================

mlflow.set_experiment("House_Price_Prediction")

# =========================================================
# Run preprocessing
# =========================================================

preprocess_data()

# =========================================================
# Load processed data
# =========================================================

X_train, X_test, y_train, y_test = load_processed_data()

# =========================================================
# Models Dictionary
# =========================================================

models = {

    "LinearRegression": LinearRegression(),

    "DecisionTree": DecisionTreeRegressor(
        max_depth=10,
        random_state=42
    ),

    "RandomForest": RandomForestRegressor(
        n_estimators=30,
        max_depth=8,
        random_state=42
    ),

    "GradientBoosting": GradientBoostingRegressor(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=5,
        random_state=42
    ),

    "XGBoost": XGBRegressor(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=5,
        random_state=42,
        objective="reg:squarederror"
    )
}

# =========================================================
# Variables to track best model
# =========================================================

best_score = -999
best_model = None
best_model_name = ""

# =========================================================
# Create models folder
# =========================================================

os.makedirs("../models", exist_ok=True)

# =========================================================
# Train Models
# =========================================================

for model_name, model in models.items():

    with mlflow.start_run(run_name=model_name):

        # -------------------------------------------------
        # Train model
        # -------------------------------------------------

        model.fit(X_train, y_train)

        # -------------------------------------------------
        # Predictions
        # -------------------------------------------------

        predictions = model.predict(X_test)

        # -------------------------------------------------
        # Metrics
        # -------------------------------------------------

        r2 = r2_score(y_test, predictions)

        mae = mean_absolute_error(y_test, predictions)

        mse = mean_squared_error(y_test, predictions)

        rmse = mse ** 0.5

        # -------------------------------------------------
        # Print Results
        # -------------------------------------------------

        print("\n===================================")
        print(f"Model : {model_name}")
        print(f"R2 Score : {r2}")
        print(f"MAE      : {mae}")
        print(f"RMSE     : {rmse}")
        print("===================================")

        # -------------------------------------------------
        # Log Parameters
        # -------------------------------------------------

        mlflow.log_param("model_name", model_name)

        # Log all hyperparameters
        mlflow.log_params(model.get_params())

        # -------------------------------------------------
        # Log Metrics
        # -------------------------------------------------

        mlflow.log_metric("r2_score", r2)

        mlflow.log_metric("mae", mae)

        mlflow.log_metric("mse", mse)

        mlflow.log_metric("rmse", rmse)

        # -------------------------------------------------
        # Add Tags
        # -------------------------------------------------

        mlflow.set_tag(
            "project",
            "House Price Prediction"
        )

        mlflow.set_tag(
            "developer",
            "Aishwarya Gahukar"
        )

        # -------------------------------------------------
        # Log Model to MLflow
        # -------------------------------------------------

        mlflow.sklearn.log_model(
            sk_model=model,
            name=model_name
        )

        # -------------------------------------------------
        # Track Best Model
        # -------------------------------------------------

        if r2 > best_score:

            best_score = r2

            best_model = model

            best_model_name = model_name

# =========================================================
# Save Best Model
# =========================================================

joblib.dump(
    best_model,
    "../models/model.pkl",
    compress=3
)

# =========================================================
# Save Best Model Name
# =========================================================

with open("../models/best_model.txt", "w") as f:

    f.write(best_model_name)

# =========================================================
# Save Scaler
# =========================================================

mlflow.log_artifact("../models/scaler.pkl")

# =========================================================
# Final Output
# =========================================================

print("\n===================================")
print(f"Best Model : {best_model_name}")
print(f"Best Score : {best_score}")
print("===================================")

print("\nBest model saved successfully")