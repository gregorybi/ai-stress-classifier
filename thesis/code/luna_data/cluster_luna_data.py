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


# 1 Read the Excel File: 
# Replace 'your_file.xlsx' with the actual path to your Excel file
excel_file_path = '/home/gb/thesis/code/luna_data/luna_data.ods'
df = pd.read_excel(excel_file_path)  # Read the entire Excel file

# 2 Data Preparation:

mean_values = list(df['Mean(μS)'])
rise_time = list(df['Rise Time (sec)'])
peak_ampl = list(df['Peak Amplitude (μS)'])
state = list(df['State'])

# Create a KMeans model with the desired number of clusters
n_clusters = 4  # You can adjust this based on your data
kmeans = KMeans(n_clusters=n_clusters)

# list of lists for the classifier
X = [] 

for i in range(len(state)) :
    X.append([mean_values[i], peak_ampl[i]])

#predict the labels of clusters.
label = kmeans.fit_predict(X)

print(label)

for i in range(len(X)):
    if label[i] == 0:
        plt.scatter(X[i][0], X[i][1], color='blue')
    elif label[i] == 1:
        plt.scatter(X[i][0], X[i][1], color='red')
    elif label[i] == 2:
        plt.scatter(X[i][0], X[i][1], color='green')
    else :             
        plt.scatter(X[i][0], X[i][1], color='black')


plt.show()