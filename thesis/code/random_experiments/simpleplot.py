import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm

# Example GSR data (replace with your actual data)
gsr_data = [0.8, 0.6, 0.9, 0.5, 0.7, 0.4, 0.8]

# Define a threshold (adjust as needed)
threshold = 0.7

# Create labels based on the threshold
labels = [1 if gsr > threshold else 0 for gsr in gsr_data]  # 1 for stressed, 0 for relaxed

# Convert data and labels to numpy arrays
X = np.array(gsr_data).reshape(-1, 1)
y = np.array(labels)

# Initialize an SVM classifier (linear kernel)
clf = svm.SVC(kernel="linear")

# Train the classifier
clf.fit(X, y)

# Predict new data points (e.g., test_data)
test_data = [0.75, 0.3, 0.85]
predictions = clf.predict(np.array(test_data).reshape(-1, 1))

# Create a scatter plot
plt.figure(figsize=(8, 6))
plt.scatter(X, y, c=y, cmap="coolwarm", label="Data points", marker="o")
plt.scatter(test_data, predictions, c="black", marker="x", s=100, label="Test points")
plt.axhline(y=threshold, color="gray", linestyle="--", label="Threshold")

plt.xlabel("GSR Value")
plt.ylabel("Classification")
plt.title("GSR Data Classification")
plt.legend()
plt.grid(True)
plt.show()
