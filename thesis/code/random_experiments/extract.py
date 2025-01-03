import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report


# 1 Read the Excel File: 
# Replace 'your_file.xlsx' with the actual path to your Excel file
excel_file_path = 'thesis/code/GSR_uS_tests_FinalVersion.ods'
df = pd.read_excel(excel_file_path)  # Read the entire Excel file
#print(df)


# 2 Data Preparation:
# Assuming your Excel file has a column named “GSR” containing the GSR measurements, extract that column:

gsr_values = df['test16Y']
#new_values = gsr_values['Vx (Volt)']

print(df)
print(gsr_values)
#print(new_values)
