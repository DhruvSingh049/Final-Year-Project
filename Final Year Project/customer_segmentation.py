import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Load the dataset
data = pd.read_csv('Mall_Customers.csv')

# Select relevant features for segmentation
features = ['Age', 'Annual Income (k$)', 'Spending Score (1-100)']

# Extract features for segmentation
X = data[features]

# Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Choose the number of clusters (you may need to experiment with this)
num_clusters = 3

# Train k-means clustering model
kmeans = KMeans(n_clusters=num_clusters, random_state=42)
data['segment'] = kmeans.fit_predict(X_scaled)

# Visualize the clusters in 3D
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(X_scaled[:, 0], X_scaled[:, 1], X_scaled[:, 2], c=data['segment'], cmap='viridis', s=50)

ax.set_xlabel('Age')
ax.set_ylabel('Annual Income (k$)')
ax.set_zlabel('Spending Score (1-100)')
ax.set_title('Customer Segmentation')

plt.show()

# Analyze the results and tailor strategies for each segment
segmentation_analysis = data.groupby('segment').mean()

# Print the results
print(segmentation_analysis)
