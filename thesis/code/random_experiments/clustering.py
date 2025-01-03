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

gsr_values = np.array(gsr_data).reshape(-1, 1)  # Reshape to a 2D array

# Create a KMeans model with the desired number of clusters
n_clusters = 2  # You can adjust this based on your data
kmeans = KMeans(n_clusters=n_clusters)

# Fit the model to your GSR values
kmeans.fit(gsr_values)

# Get cluster assignments for each data point
cluster_labels = kmeans.labels_
stress_relax_labels = ['relaxed' if label == 0 else 'stressed' for label in cluster_labels]

# Print the cluster labels
print("Cluster labels:", stress_relax_labels)

# Create a scatter plot
plt.figure(figsize=(8, 6))
scatter = plt.scatter(range(len(gsr_values)), gsr_values, c=cluster_labels, cmap='viridis', marker='o')

legend_colors = scatter.to_rgba([0, 1])  # Assuming 0 corresponds to relaxed and 1 corresponds to stressed

# Add custom legend entries with the same colors
#plt.scatter([], [], color=legend_colors[0], marker='o', label="Relaxed")
#plt.scatter([], [], color=legend_colors[1], marker='o', label="Stressed")


plt.xlabel("Data Point Index")
plt.ylabel("GSR Value")
plt.title("K-Means Clustering of GSR Values")
#plt.colorbar(label="Cluster Label")
plt.legend()

# Show the plot
plt.show()