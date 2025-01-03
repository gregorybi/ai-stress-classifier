import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report


# 1 Read the Excel File: 
# Replace 'your_file.xlsx' with the actual path to your Excel file
excel_file_path = 'your_file.xlsx'
df = pd.read_excel(excel_file_path)  # Read the entire Excel file

# 2 Data Preparation:
# Assuming your Excel file has a column named “GSR” containing the GSR measurements, extract that column:

gsr_values = df['GSR']

# 3 Label Data (threshold for anxiety)

threshold = 0.5  # Adjust as needed
df['Anxiety_Label'] = df['GSR'].apply(lambda x: 'stressed' if x > threshold else 'relaxed')


# 4 Train SVM classifier
# Load your GSR data from the DataFrame (replace with actual data)

X = df[['GSR']].values
y = np.where(df['Anxiety_Label'] == 'stressed', 1, 0)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize SVM classifier
svm_classifier = SVC(kernel='linear')

# Train the classifier
svm_classifier.fit(X_train, y_train)

# Make predictions
y_pred = svm_classifier.predict(X_test)

# Evaluate performance
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

print(f"Accuracy: {accuracy:.2f}")
print("Classification Report:\n", report)
