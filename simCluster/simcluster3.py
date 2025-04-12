import pandas as pd
import torch
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv
from torch_geometric.data import Data
from sklearn.cluster import KMeans
import os

# Load Data
data = pd.read_csv("data.csv")

# Map user IDs and groupTags to unique indices
user_map = {u: i for i, u in enumerate(data['userid'].unique())}
group_map = {g: i + len(user_map) for i, g in enumerate(data['groupTag'].unique())}

data['user_idx'] = data['userid'].map(user_map)
data['group_idx'] = data['groupTag'].map(group_map)

# Create Edge Index (Graph Structure)
edges = torch.tensor(data[['user_idx', 'group_idx']].values.T, dtype=torch.long)
weights = torch.tensor(data['groupImpression'].values, dtype=torch.float)

# Create PyTorch Geometric Data Object
graph_data = Data(edge_index=edges, edge_attr=weights, num_nodes=len(user_map) + len(group_map))

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
model = GraphSAGE(in_channels=1, hidden_channels=32, out_channels=16)
x = torch.ones((graph_data.num_nodes, 1))  # Node features (dummy 1s)

# Forward Pass to Generate Embeddings
with torch.no_grad():
    embeddings = model(x, graph_data.edge_index)

# Apply K-Means Clustering
num_clusters = 5
kmeans = KMeans(n_clusters=num_clusters, random_state=42)
clusters = kmeans.fit_predict(embeddings.numpy())

# Assign Clusters
user_clusters = {user: clusters[idx] for user, idx in user_map.items()}
data['cluster'] = data['userid'].map(user_clusters)

# Save Embeddings and Clusters
os.makedirs('batch_output', exist_ok=True)
torch.save(model.state_dict(), 'batch_output/graphsage_model.pth')
pd.DataFrame(embeddings.numpy()).to_csv('batch_output/node_embeddings.csv', index=False)
data[['userid', 'cluster']].to_csv('batch_output/user_clusters.csv', index=False)

print("Batch Processing Completed. Results saved.")
