import pandas as pd
import torch
import torch.nn.functional as F
from torch_geometric_temporal.nn.recurrent import TGN
from torch_geometric_temporal.signal import StaticGraphTemporalSignal
import numpy as np
import faiss
import redis

# Step 1: Load Candidates from 12-hour Batch
num_candidates = 15000  # Subset from millions
num_users = 10000  # Active users interacting in the last hour
num_interactions = 50000  # New interactions in the past hour

# Simulate candidate embeddings from 12-hour batch
candidate_embeddings = np.random.rand(num_candidates, 16).astype('float32')
faiss_index = faiss.IndexFlatL2(16)
faiss_index.add(candidate_embeddings)  # Store embeddings for fast retrieval

# Step 2: Load Latest Interactions (Hourly Batch)
data = {
    "user_id": np.random.randint(0, num_users, num_interactions),
    "video_id": np.random.randint(0, num_candidates, num_interactions),
    "timestamp": np.random.randint(1, 1000, num_interactions),  # Simulated timestamps
    "interaction_type": np.random.choice(["watch", "like", "share"], num_interactions)
}
df = pd.DataFrame(data)

df.to_csv("user_video_interactions.csv", index=False)

# Convert to Temporal Graph Format
edge_index = torch.tensor([df['user_id'].values, df['video_id'].values], dtype=torch.long)
edge_attr = torch.tensor(df['timestamp'].values, dtype=torch.float).unsqueeze(1)
nodes_features = torch.rand((num_users + num_candidates, 16))  # Random 16-dim node features
targets = torch.randint(0, 2, (num_interactions,))

graph_data = StaticGraphTemporalSignal(
    edge_indices=[edge_index],
    edge_weights=[edge_attr],
    features=[nodes_features],
    targets=[targets]
)

# Step 3: Define Optimized TGN Model
class OptimizedTGN(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OptimizedTGN, self).__init__()
        self.recurrent = TGN(in_channels, out_channels)
        self.linear = torch.nn.Linear(out_channels, 16)  # Output refined embeddings

    def forward(self, x, edge_index, edge_weight):
        h = self.recurrent(x, edge_index, edge_weight)
        return self.linear(h)

# Step 4: Incrementally Update TGN Embeddings
model = OptimizedTGN(16, 32)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

for epoch in range(5):  # Reduce training time
    optimizer.zero_grad()
    output = model(graph_data.features[0], edge_index, edge_attr)
    loss = F.mse_loss(output, graph_data.features[0])  # Self-supervised learning
    loss.backward()
    optimizer.step()
    print(f"Epoch {epoch + 1}, Loss: {loss.item()}")

# Step 5: Use Updated TGN Embeddings for Candidate Refinement
with torch.no_grad():
    updated_embeddings = model.recurrent(nodes_features, edge_index, edge_attr).numpy()

    # Find top-K nearest refined candidates for each user
    k = 100  # Reduce from 15K to top 100 per user
    refined_candidates = []
    for user_id in range(num_users):
        _, top_k_indices = faiss_index.search(updated_embeddings[user_id].reshape(1, -1), k)
        refined_candidates.append((user_id, top_k_indices[0].tolist()))

# Step 6: Store Candidates in Redis for Real-Time Retrieval
redis_client = redis.StrictRedis(host='localhost', port=6379, db=0)
for user_id, candidates in refined_candidates:
    redis_client.set(f"user:{user_id}:candidates", str(candidates))

print("Updated candidates stored in Redis for real-time ranking.")
