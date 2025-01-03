import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt



# 1 Read the Excel File: 
# Replace 'your_file.xlsx' with the actual path to your Excel file
excel_file_path = '/home/gb/thesis/code/av/Self-annotation_Single_Modal.ods'
df = pd.read_excel(excel_file_path)  # Read the entire Excel file
#print(df)


# 2 Data Preparation:

valence_values = list(df['Valence Preprocessed'])
arousal_values = list(df['Arousal Preprocessed'])
dominance_values = list(df['Dominance Preprocessed'])


# list for the av values as coordinates
coordinates = []

# print("Valence Values: ", valence_values)
# print ("Arousal", arousal_values)
# print ("Dominance", dominance_values)

fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)

# Move left y-axis and bottom x-axis to the center, passing through (0,0)
ax.spines['left'].set_position('center')
ax.spines['bottom'].set_position('center')

# Eliminate upper and right axes
ax.spines['right'].set_color('none')
ax.spines['top'].set_color('none')

# Show ticks in the left and lower axes only
ax.xaxis.set_ticks_position('bottom')
ax.yaxis.set_ticks_position('left')

#show all arousal and valence values
for i in range(len(valence_values)) :
    ax.scatter (valence_values[i], arousal_values[i], marker='o')
    

# classification

    




plt.xlabel('Valence')
plt.ylabel('Arousal')

plt.show()


