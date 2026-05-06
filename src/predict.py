import joblib
import numpy as np

model = joblib.load("models/model.pkl")

sample_data = np.array([[8.3252,
                         41,
                         6.984,
                         1.023,
                         322,
                         2.555,
                         37.88,
                         -122.23]])

prediction = model.predict(sample_data)

print("Predicted Price:", prediction)