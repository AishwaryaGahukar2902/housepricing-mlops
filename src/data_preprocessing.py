import os
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib


def preprocess_data():

    # Load dataset
    df = pd.read_csv("../data/housing.csv")

    # Features and target
    X = df.drop("PRICE", axis=1)
    y = df["PRICE"]

    # Train test split
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    # Scaling
    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Create processed folder
    os.makedirs("../data/processed", exist_ok=True)

    # Save processed datasets
    pd.DataFrame(X_train_scaled).to_csv(
        "../data/processed/X_train.csv",
        index=False
    )

    pd.DataFrame(X_test_scaled).to_csv(
        "../data/processed/X_test.csv",
        index=False
    )

    pd.DataFrame(y_train).to_csv(
        "../data/processed/y_train.csv",
        index=False
    )

    pd.DataFrame(y_test).to_csv(
        "../data/processed/y_test.csv",
        index=False
    )

    # Save scaler
    os.makedirs("../models", exist_ok=True)

    joblib.dump(
        scaler,
        "../models/scaler.pkl"
    )

    print("Preprocessing completed")
    print("Processed files saved")



def load_processed_data():

    X_train = pd.read_csv("../data/processed/X_train.csv")
    X_test = pd.read_csv("../data/processed/X_test.csv")

    y_train = pd.read_csv("../data/processed/y_train.csv").values.ravel()
    y_test = pd.read_csv("../data/processed/y_test.csv").values.ravel()

    return X_train, X_test, y_train, y_test


if __name__ == "__main__":
    preprocess_data()