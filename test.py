from model import deep_neural_network, predict
import numpy as np
from sklearn import datasets, metrics
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Load digits dataset
digits = datasets.load_digits()

# Flatten the images
n_samples = len(digits.images)
data = digits.images.reshape((n_samples, -1))

# Split data into 70% train and 30% test subsets
X_train, X_test, y_train, y_test = train_test_split(
    data, digits.target, test_size=0.3, shuffle=False
)

# Normalize input data
scaler = StandardScaler()
X_train_normalized = scaler.fit_transform(X_train)
X_test_normalized = scaler.transform(X_test)

# Train your model
training_history = deep_neural_network(X_train_normalized.T, y_train.reshape(1, -1))

# Make predictions on the test set
predictions_test = predict(X_test_normalized.T, training_history[-1]['parametres'])

# Evaluate the accuracy
accuracy = metrics.accuracy_score(y_test, predictions_test.flatten())
print("Test Accuracy:", accuracy)