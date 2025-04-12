import pandas as pd
import torch
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv
from torch_geometric.data import Data
from sklearn.mixture import GaussianMixture
import numpy as np
import time

# Load Data
INTERACTION_FILE = 'interaction_data.csv'
CONTENT_FILE = 'content_metadata.csv'
USER_FILE = 'user_metadata.csv'

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
model = GraphSAGE(in_channels=node_features.shape[1], hidden_channels=128, out_channels=64)
x = node_features

# Train Model
with torch.no_grad():
    embeddings = model(x, graph_data.edge_index)

# Apply Gaussian Mixture Model (Soft Clustering)
num_clusters = 100
gmm = GaussianMixture(n_components=num_clusters, random_state=42)
gmm.fit(embeddings.numpy())

# Extract Soft Cluster Probabilities
probabilities = gmm.predict_proba(embeddings.numpy())

# Map user soft clusters
user_cluster_probabilities = {user: probabilities[user_map[user]] for user in user_map}

# Apply Recency Boost
def recency_boost(content_id):
    content_age = (time.time() - content_data[content_data['content_id'] == content_id]['upload_time'].values[0]) / 3600
    if content_age <= 12:
        return 2.0
    elif content_age <= 24:
        return 1.5
    else:
        return 1.0

# Apply Popularity Decay
def popularity_decay(content_id):
    popularity = content_data[content_data['content_id'] == content_id]['view_count'].values[0]
    decay_factor = np.log10(popularity + 1)
    if decay_factor > 5:
        return 0.5
    elif decay_factor > 3:
        return 0.8
    else:
        return 1.0

# Candidate Generation Function
def generate_candidates_for_user(user_id):
    if user_id not in user_map:
        print(f"User ID {user_id} not found.")
        return

    cluster_probs = user_cluster_probabilities[user_id]
    top_clusters = np.argsort(cluster_probs)[::-1][:5]  # Top 5 clusters per user

    top_content = []
    for cluster in top_clusters:
        cluster_content = merged_data[merged_data['cluster'] == cluster]['content_id'].value_counts().index.tolist()
        for content in cluster_content:
            boost = recency_boost(content)
            decay = popularity_decay(content)
            final_score = boost * decay
            top_content += [content] * int(final_score * 10)

    # Add random content for diversity
    random_content = merged_data['content_id'].sample(2000).tolist()

    # Combine top cluster content + random content
    candidates = list(set(top_content[:4000] + random_content))
    print(f"Generated Candidates for User {user_id}: {candidates}")
    return candidates

# Example usage
user_id_example = list(user_map.keys())[0]  # Replace with any user_id
generate_candidates_for_user(user_id_example)
