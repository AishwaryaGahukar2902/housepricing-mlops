from sklearn.datasets import fetch_california_housing
import pandas as pd
import os

def load_data():
    housing = fetch_california_housing()

    df = pd.DataFrame(housing.data, columns=housing.feature_names)

    df["PRICE"] = housing.target

    os.makedirs("data", exist_ok=True)

    df.to_csv("C:/Users/hp/house-price-mlops/housing.csv", index=False)

    print("Dataset saved successfully")

if __name__ == "__main__":
    load_data()