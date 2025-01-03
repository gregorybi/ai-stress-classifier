import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import numpy as np
import math
from sklearn.cluster import KMeans

"""
Plot1 : All the sample points
Plot2 : Training and testing points
"""


# 1 Read the Excel File: 
# Replace 'your_file.xlsx' with the actual path to your Excel file
excel_file_path = '/home/gb/thesis/code/luna_data/luna_data.ods'
df = pd.read_excel(excel_file_path)  # Read the entire Excel file

# 2 Data Preparation:

mean_values = list(df['Mean(μS)'])
rise_time = list(df['Rise Time (sec)'])
peak_ampl = list(df['Peak Amplitude (μS)'])
state = list(df['State'])

#list of lists for SVM
X = [] 
y = state

print(mean_values)
print(rise_time)

for i in range(len(state)) :
    X.append([mean_values[i], peak_ampl[i]])

print('X is: ', X)


# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("y_test: ", y_test)

# Scale features (preprocessing)
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# Initialize SVM model
svm_model = SVC(kernel='rbf', C=1.0)
svm_model.fit(X_train, y_train)

# Predictions
y_pred = svm_model.predict(X_test)
print("y_pred: ", y_pred)


# Evaluate
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")

# plot all the sample points
plt.subplot(1, 2, 1)

plt.title("Sample Points")

for i in range(len(state)) :

    if state[i] == 'Stressed':
        plt.scatter(mean_values[i], peak_ampl[i], color='r', marker='o')
    else : 
        plt.scatter(mean_values[i], peak_ampl[i], color='b', marker='o')


plt.scatter([], [], color='red', marker='o', label='stressed')
plt.scatter([], [], color='blue', marker='o', label='relaxed')

plt.legend()
plt.xlabel('Mean Values')
plt.ylabel('Peak Amplitude')


# plot training and testing points
plt.subplot(1, 2, 2)
plt.title('Training and Testting points')


for i in range(len(X_train_scaled)) :
    if y_train[i] == 'Stressed':
        plt.scatter(X_train_scaled[i][0], X_train_scaled[i][1], c = 'red', edgecolors='k', s=50)
    else:
        plt.scatter(X_train_scaled[i][0], X_train_scaled[i][1], c = 'blue', edgecolors='k', s=50)

for i in range(len(X_test_scaled)) :
    if y_pred[i] == 'Stressed':
        plt.scatter(X_test_scaled[i][0], X_test_scaled[i][1], c = 'red', s=50, marker='x')
    else:
        plt.scatter(X_test_scaled[i][0], X_test_scaled[i][1], c = 'blue', s=50, marker='x')

plt.scatter([], [], color='red', marker='x', label='stressed test point')
plt.scatter([], [], color='blue', marker='x', label='relaxed test point')
plt.scatter([], [], color='red', marker='o', label='stressed train point')
plt.scatter([], [], color='blue', marker='o', label='relaxed train point')

plt.legend()

plt.tight_layout()
plt.show()



