import pandas as pd
import torch
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv
from torch_geometric.data import Data
from sklearn.mixture import GaussianMixture
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import time

# Load Data
interaction_data = pd.read_csv('interaction_data.csv')
content_data = pd.read_csv('content_metadata.csv')
user_data = pd.read_csv('user_metadata.csv')

# Map user IDs and content IDs to unique indices
user_map = {u: i for i, u in enumerate(user_data['userid'].unique())}
content_map = {c: i + len(user_map) for i, c in enumerate(content_data['content_id'].unique())}

interaction_data['user_idx'] = interaction_data['userid'].map(user_map)
interaction_data['content_idx'] = interaction_data['content_id'].map(content_map)

# Normalize engagement scores
interaction_data['engagement_score'] = np.log1p(interaction_data['engagement_score'])

# Create Edge Index
edges = torch.tensor(interaction_data[['user_idx', 'content_idx']].values.T, dtype=torch.long)
weights = torch.tensor(interaction_data['engagement_score'].values, dtype=torch.float)

# Convert user and content metadata to features
user_data_encoded = pd.get_dummies(user_data.drop(['userid'], axis=1)).astype(float)
user_features = torch.tensor(user_data_encoded.values, dtype=torch.float)
content_data_encoded = pd.get_dummies(content_data.drop(['content_id', 'title'], axis=1)).astype(float)
content_features = torch.tensor(content_data_encoded.values, dtype=torch.float)

# Ensure dimensions match
min_dim = min(user_features.shape[1], content_features.shape[1])
user_features = user_features[:, :min_dim]
content_features = content_features[:, :min_dim]

# Combine features
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

# Train Model (Add fine-tuning if needed)
with torch.no_grad():
    embeddings = model(x, graph_data.edge_index)

# Apply Gaussian Mixture Model (Soft Clustering)
num_clusters = 100
gmm = GaussianMixture(n_components=num_clusters, random_state=42)
gmm.fit(embeddings.numpy())

# Extract Soft Cluster Probabilities
probabilities = gmm.predict_proba(embeddings.numpy())
user_cluster_probabilities = {user: probabilities[user_map[user]] for user in user_map}

# Candidate Generation Function
def generate_candidates_for_user(user_id, num_candidates=100):
    if user_id not in user_map:
        print(f"User ID {user_id} not found.")
        return []

    # Get user embedding
    user_idx = user_map[user_id]
    user_embedding = embeddings[user_idx].numpy().reshape(1, -1)

    # Get content indices
    content_indices = list(content_map.values())
    content_embeddings = embeddings[content_indices].numpy()

    # Compute cosine similarity
    similarities = cosine_similarity(user_embedding, content_embeddings)[0]

    # Rank by similarity and engagement
    ranked_content = sorted(zip(content_indices, similarities), key=lambda x: x[1], reverse=True)

    # Convert indices back to content IDs
    candidates = [list(content_map.keys())[list(content_map.values()).index(idx)] for idx, _ in ranked_content[:num_candidates]]

    print(f"Generated Candidates for User {user_id}: {candidates}")
    return candidates

# Example usage
user_id_example = list(user_map.keys())[0]
generate_candidates_for_user(user_id_example, num_candidates=20)
generate_candidates_for_user('user_401', num_candidates=20)
generate_candidates_for_user('user_100', num_candidates=20)