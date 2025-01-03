import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans

# Load your GSR data from the spreadsheet 
# Replace 'your_data.ods' with the actual file path
data = pd.read_excel('/home/gb/thesis/code/GSR_only.ods')
gsr_data = []

# Calculate the difference between max and min GSR measurements for each individual
#data['gsr_difference'] = data.max(axis=1) - data.min(axis=1)
for col in data.columns:
     # Calculate the maximum difference for the current column
        max_diff = data[col].max() - data[col].min()
        # Assign the maximum difference to the corresponding row in the 'difference' column
        # data.loc[:, 'gsr_difference'] = max_diff
        gsr_data.append(max_diff)

gsr_values = np.array(gsr_data).reshape(-1, 1)  # Reshape to a 2D array

print(gsr_data)
print()
print(gsr_values)


# Create a KMeans model with the desired number of clusters
n_clusters = 2  # You can adjust this based on your data
kmeans = KMeans(n_clusters=n_clusters)

# Fit the model to your GSR values

gsr_new_values = MinMaxScaler().fit_transform(gsr_values)
clustered = kmeans.fit(gsr_new_values)
# Get cluster assignments for each data point
cluster_labels = kmeans.labels_


for index, value in enumerate(gsr_data):
        if value == max(gsr_data) :
                marker = index #mark the index of a definetely stressed value
                break
        
print("marker=" , marker)

cluster_labels = list (cluster_labels)

stress_relax_labels = []

for i in cluster_labels:
    stress_relax_labels.append(str(i))



print("cluster_labels[marker]= " ,cluster_labels[marker])
print ("type of marker: ", type(cluster_labels[marker]))


if cluster_labels[marker] == 0 :
    print("is 0")
    for j in range(len(stress_relax_labels)):
        if stress_relax_labels[j]== '0' :
            stress_relax_labels[j] = 'stressed'
        else :
             stress_relax_labels[j] = 'relaxed'
else :
    print("is 1")
    for j in range(len (stress_relax_labels)):
        if stress_relax_labels[j] == '1':
            stress_relax_labels[j] = 'stressed'
        else:
             stress_relax_labels[j] = 'relaxed'



# Print the cluster labels
print("Cluster labels:", cluster_labels)
print("Stress relaxed labels:", stress_relax_labels)

#SVM Begins 

X = [[gsr] for gsr in gsr_data]  # Create a list of lists (each inner list contains one GSR value)
y = stress_relax_labels

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


# Clusterring subplot
plt.subplot(1,2,1)
scatter = plt.scatter(range(len(gsr_values)), gsr_new_values, c=cluster_labels, cmap='viridis', marker='o')

legend_colors = scatter.to_rgba([0, 1])  # Assuming 0 corresponds to relaxed and 1 corresponds to stressed

# Add custom legend entries with the same colors
# Associate 0 with always relaxed, 1 with always stressed

if cluster_labels[marker] == 0 :
    plt.scatter([], [], color=legend_colors[1], marker='o', label="Relaxed")
    plt.scatter([], [], color=legend_colors[0], marker='o', label="Stressed")
else:
     plt.scatter([], [], color=legend_colors[0], marker='o', label="Relaxed")
     plt.scatter([], [], color=legend_colors[1], marker='o', label="Stressed")



plt.xlabel("Data Point Index")
plt.ylabel("Normalized GSR Value")
plt.title("K-Means Clustering of GSR Values")
plt.legend()
#plt.colorbar(label="Cluster Label")



#SVM subplot
plt.subplot(1,2,2)
plt.scatter(X_train_scaled, y_train, c='black', label="Data points", edgecolors='k', s=50)
plt.scatter(X_test_scaled, y_test, marker='x', s=100, label="Test points")

plt.xlabel('GSR Feature')
plt.ylabel('Class (Relaxed or Stressed)')
plt.title('SVM Classification')
plt.legend()
plt.grid(True)


plt.tight_layout()
plt.show()