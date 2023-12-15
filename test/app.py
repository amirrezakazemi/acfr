import joblib
import numpy as np

# Load the trained machine learning model
model = joblib.load('your_model.pkl')

def predict(input_data):
    # Assuming input_data is a NumPy array
    prediction = model.predict(input_data)
    return prediction.tolist()

if __name__ == "__main__":

    
    # Example usage
    input_data = np.array([[1, 2, 3, 4]])  # Adjust based on your model input requirements
    result = predict(input_data)
    print(f"Prediction: {result}")
