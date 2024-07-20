import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import joblib

# Sample data (replace with your actual dataset)
X = np.array([[3, 7], [4, 9], [2, 5], [6, 8], [3, 6], [5, 7], [8, 6], [7, 9], [4, 5], [6, 7]])
y = np.array(['happy', 'sad', 'neutral', 'angry', 'anxious', 'happy', 'sad', 'angry', 'neutral', 'anxious'])

# Encode categorical labels into numerical values
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# Initialize the model (Logistic Regression) with increased max_iter
model = LogisticRegression(max_iter=1000)  # Increase max_iter to 1000 or more

# Train the model
model.fit(X_train, y_train)

# Predictions on test set
y_pred = model.predict(X_test)

# Evaluate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}")

# Save the model to a file
model_filename = 'model.pkl'
joblib.dump(model, model_filename)
print(f"Model saved to {model_filename}")

# Example of predicting a new input
new_data = np.array([[5, 7]])  # Example input data (activity level = 5, sleep quality = 7)
predicted_mood = label_encoder.inverse_transform(model.predict(new_data))
print(f"Predicted mood for new data: {predicted_mood[0]}")
