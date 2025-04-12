import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# Load data
data = pd.read_csv("../data.csv")

# Normalize groupImpression using Min-Max scaling
scaler = MinMaxScaler()
data["normalizedImpression"] = scaler.fit_transform(data[["groupImpression"]])

print(data.head())  # Check the normalized values

import networkx as nx

# Create an undirected graph
G = nx.Graph()

# Add edges (user <-> groupTag) with normalized impressions as edge weights
for _, row in data.iterrows():
    G.add_edge(row["userid"], row["groupTag"], weight=row["normalizedImpression"])

print("Graph nodes:", G.number_of_nodes())  # Check the graph structure
print("Graph edges:", G.number_of_edges())


from node2vec import Node2Vec

# Generate embeddings using Node2Vec
node2vec = Node2Vec(G, dimensions=16, walk_length=10, num_walks=50, workers=4)
model = node2vec.fit(window=5, min_count=1, batch_words=4)

# Get embeddings for all nodes (users & groups)
embeddings = {node: model.wv[node] for node in G.nodes}

# Convert to a DataFrame for clustering
import numpy as np
embedding_matrix = np.array([embeddings[node] for node in G.nodes])

from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=5, random_state=42)
clusters = kmeans.fit_predict(embedding_matrix)

# Assign clusters to users
user_cluster_map = dict(zip(G.nodes, clusters))
data["cluster"] = data["userid"].map(user_cluster_map)

print(data[["userid", "cluster"]].drop_duplicates())

def recommend_groups(user_id, data):
    user_cluster = data[data["userid"] == user_id]["cluster"].values[0]
    similar_users = data[data["cluster"] == user_cluster]["userid"].unique()

    # Find most frequent groupTags in the cluster
    recommended_groups = data[data["userid"].isin(similar_users)]["groupTag"].value_counts().index.tolist()

    return recommended_groups[:3]  # Top 3 recommendations

print("Recommended Groups:", recommend_groups("7u6t5r4s3q", data))
