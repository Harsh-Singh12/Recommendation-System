import pandas as pd
import torch
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv
from torch_geometric.data import Data
from sklearn.cluster import KMeans
import os
from flask import Flask, request, jsonify
import time

app = Flask(__name__)

# Load Data
INTERACTION_FILE = 'interaction_data.csv'
CONTENT_FILE = 'content_metadata.csv'
USER_FILE = 'user_metadata.csv'

# Load Data
interaction_data = pd.read_csv(INTERACTION_FILE)
content_data = pd.read_csv(CONTENT_FILE)
user_data = pd.read_csv(USER_FILE)

# Merge all data
merged_data = interaction_data.merge(content_data, on='content_id')
merged_data = merged_data.merge(user_data, on='userid')

# Map user IDs and content IDs to unique indices
user_map = {u: i for i, u in enumerate(merged_data['userid'].unique())}
content_map = {c: i + len(user_map) for i, c in enumerate(merged_data['content_id'].unique())}

merged_data['user_idx'] = merged_data['userid'].map(user_map)
merged_data['content_idx'] = merged_data['content_id'].map(content_map)

# Create Edge Index
edges = torch.tensor(merged_data[['user_idx', 'content_idx']].values.T, dtype=torch.long)
weights = torch.tensor(merged_data['engagement_score'].values, dtype=torch.float)

# Combine metadata as node features
user_features = torch.tensor(user_data.drop(['userid'], axis=1).values, dtype=torch.float)
content_features = torch.tensor(content_data.drop(['content_id'], axis=1).values, dtype=torch.float)

node_features = torch.cat([user_features, content_features], dim=0)

# Create Graph Data
graph_data = Data(edge_index=edges, edge_attr=weights, x=node_features)

# Define GraphSAGE Model
class GraphSAGE(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = SAGEConv(in_channels, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, out_channels)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        return x

# Initialize Model
model = GraphSAGE(in_channels=node_features.shape[1], hidden_channels=64, out_channels=32)
x = node_features

# Train Model
with torch.no_grad():
    embeddings = model(x, graph_data.edge_index)

# Apply K-Means Clustering
num_clusters = 5
kmeans = KMeans(n_clusters=num_clusters, random_state=42)
clusters = kmeans.fit_predict(embeddings.numpy())

# Map user clusters
user_clusters = {user: clusters[user_map[user]] for user in user_map}

# Save Clusters
pd.DataFrame({'userid': list(user_clusters.keys()), 'cluster': list(user_clusters.values())}).to_csv('user_clusters.csv', index=False)

@app.route('/recommend', methods=['GET'])
def recommend():
    user_id = request.args.get('user_id')
    if user_id not in user_clusters:
        return jsonify({"error": "User ID not found"}), 404

    # Get cluster of the user
    cluster_id = user_clusters[user_id]

    # Recommend top content from the same cluster
    recommended_content = merged_data[merged_data['cluster'] == cluster_id]['content_id'].value_counts().index.tolist()[:5]

    return jsonify({"user_id": user_id, "recommended_content": recommended_content})

if __name__ == '__main__':
    app.run(port=5000)

print("SimCluster Recommendation API is ready.")
