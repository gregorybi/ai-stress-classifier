import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# Load your GSR data from the spreadsheet 
# Replace 'your_data.ods' with the actual file path
data = pd.read_excel('thesis/code/GSR_only.ods')
gsr_values = []

# Calculate the difference between max and min GSR measurements for each individual
#data['gsr_difference'] = data.max(axis=1) - data.min(axis=1)
for col in data.columns:
     # Calculate the maximum difference for the current column
        max_diff = data[col].max() - data[col].min()
        # Assign the maximum difference to the corresponding row in the 'difference' column
        # data.loc[:, 'gsr_difference'] = max_diff
        gsr_values.append(max_diff)
        print(max_diff)

gsr_values = [gsr_values]
col_names = ['gsr_difference']

ndf  = pd.DataFrame(gsr_values).T
ndf.columns = ['gsr_difference']

# Save the updated DataFrame back to a new CSV file
ndf.to_excel('updated_spreadsheet.ods', index=False)

print("Maximum differences calculated and saved to 'updated_spreadsheet.ods'.")


# Define your threshold for classification (you can adjust this)
threshold = 0.5  # Example threshold value


# Assign labels based on the threshold
ndf['label'] = ndf['gsr_difference'].apply(lambda x: 'anxious' if x > threshold else 'relaxed')

print (ndf['gsr_difference'])
print (ndf['label'])

# Split data into features (X) and labels (y)
X = ndf.drop(columns=['label', 'gsr_difference'])
y = ndf['label']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the SVM model
svm_model = SVC(kernel='linear')
svm_model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = svm_model.predict(X_test)

# Evaluate the model's accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")

# Example prediction for a new individual
#new_individual_gsr_difference = 0.6  # Replace with the actual GSR difference
#new_individual_label = svm_model.predict([[new_individual_gsr_difference]])[0]
#print(f"Predicted label for the new individual: {new_individual_label}")
