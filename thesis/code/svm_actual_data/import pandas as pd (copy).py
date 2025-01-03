import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import numpy as np

# Load your GSR data from the spreadsheet 
# Replace 'your_data.ods' with the actual file path
data = pd.read_excel('thesis/code/GSR_only.ods')
gsr_data = []

# Calculate the difference between max and min GSR measurements for each individual
#data['gsr_difference'] = data.max(axis=1) - data.min(axis=1)
for col in data.columns:
     # Calculate the maximum difference for the current column
        max_diff = data[col].max() - data[col].min()
        # Assign the maximum difference to the corresponding row in the 'difference' column
        # data.loc[:, 'gsr_difference'] = max_diff
        gsr_data.append(max_diff)

gsr_values = [gsr_data]
col_names = ['gsr_difference']

ndf  = pd.DataFrame(gsr_values).T
ndf.columns = ['gsr_difference']

# Save the updated DataFrame back to a new CSV file
ndf.to_excel('updated_spreadsheet.ods', index=False)

print("Maximum differences calculated and saved to 'updated_spreadsheet.ods'.")

# Assuming 'gsr_data' is your list of GSR values
threshold = 4


X = [[gsr] for gsr in gsr_data]  # Create a list of lists (each inner list contains one GSR value)
y = ["1" if gsr > threshold else "0" for gsr in gsr_data]

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale features (preprocessing)
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

#print(X_train_scaled)
#print(X_test_scaled)

# Initialize SVM model
svm_model = SVC(kernel='linear', C=1.0)
svm_model.fit(X_train_scaled, y_train)

# Predictions
y_pred = svm_model.predict(X_test_scaled)

# Evaluate
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")

# Generate a mesh grid along the GSR feature axis
x_min, x_max = np.min(X_train_scaled), np.max(X_train_scaled)
xx = np.linspace(x_min, x_max, 100)

# Predict class labels for each point in the mesh grid
Z = svm_model.predict(xx.reshape(-1, 1))


# Plot the decision boundary and data points
plt.figure(figsize=(8, 6))
plt.scatter(X_train_scaled, y_train, c='black', label="Data points", edgecolors='k', s=50)
plt.scatter(X_test_scaled, y_test, marker='x', s=100, label="Test points")

plt.xlabel('GSR Feature')
plt.ylabel('Class (0: Relaxed, 1: Stressed)')
plt.title('SVM Classification')
plt.legend()
plt.grid(True)
plt.show()