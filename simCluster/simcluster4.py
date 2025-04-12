import networkx as nx
import numpy as np
import faiss
import pandas as pd
from scipy.sparse.linalg import eigsh
from collections import defaultdict

class SimClusterOptimized:
    def __init__(self, num_clusters=100, embedding_dim=128):
        self.num_clusters = num_clusters
        self.embedding_dim = embedding_dim
        self.graph = nx.Graph()
        self.video_clusters = {}
        self.video_embeddings = {}

    def add_interaction(self, user, video, weight=1.0):
        """Adds an edge between user and video with a given weight."""
        self.graph.add_edge(user, video, weight=weight)

    def project_video_graph(self):
        """Projects bipartite graph to a video-video similarity graph."""
        video_graph = defaultdict(lambda: defaultdict(float))

        for user in self.graph.nodes():
            if user.startswith('user_'):
                videos = list(self.graph.neighbors(user))
                for i in range(len(videos)):
                    for j in range(i + 1, len(videos)):
                        video_graph[videos[i]][videos[j]] += 1
                        video_graph[videos[j]][videos[i]] += 1

        G_video = nx.Graph()
        for video1, neighbors in video_graph.items():
            for video2, weight in neighbors.items():
                G_video.add_edge(video1, video2, weight=weight)

        return G_video

    def compute_embeddings(self):
        """Computes low-dimensional embeddings using eigenvectors."""
        G_video = self.project_video_graph()
        nodes = list(G_video.nodes())

        if len(nodes) < 2:
            return  # Avoid errors if there are too few nodes

        # Compute adjacency matrix
        A = nx.to_numpy_array(G_video, nodelist=nodes)

        # Compute Laplacian matrix
        D = np.diag(A.sum(axis=1))
        L = D - A

        # Ensure k is valid for eigsh
        k = min(self.embedding_dim, len(nodes) - 1)
        eigvals, eigvecs = eigsh(L, k=k, which='SM')
        eigvecs = np.real(eigvecs)

        # Store embeddings
        self.video_embeddings = {nodes[i]: eigvecs[i] for i in range(len(nodes))}

    def perform_faiss_clustering(self):
        """Clusters video embeddings using FAISS for large-scale clustering."""
        videos = list(self.video_embeddings.keys())

        if not videos:
            return  # Avoid errors if no embeddings exist

        embeddings = np.array([self.video_embeddings[video] for video in videos], dtype=np.float32)
        embeddings = embeddings.reshape(len(videos), -1)  # Ensure correct FAISS format

        # FAISS clustering
        kmeans = faiss.Kmeans(embeddings.shape[1], self.num_clusters, niter=20, verbose=False)
        kmeans.train(embeddings)
        _, labels = kmeans.index.search(embeddings, 1)

        # Assign videos to clusters
        self.video_clusters = {videos[i]: int(labels[i][0]) for i in range(len(videos))}

    def get_video_candidates(self, watched_videos, top_n=10):
        """Fetches top-N candidate videos from the same cluster as watched videos."""
        candidate_videos = set()
        for video in watched_videos:
            if video in self.video_clusters:
                cluster_id = self.video_clusters[video]
                candidates = [v for v, c in self.video_clusters.items() if c == cluster_id]
                candidate_videos.update(candidates)

        return list(candidate_videos)[:top_n]

    def load_interactions_from_csv(self, file_path):
        """Loads interactions from a CSV file and adds edges to the graph."""
        df = pd.read_csv(file_path)
        for _, row in df.iterrows():
            user = row['userid']
            video = row['content_id']
            weight = row['engagement_score']  # Using engagement score as edge weight
            self.add_interaction(user, video, weight)

# Load interactions from CSV
sim_cluster = SimClusterOptimized(num_clusters=50, embedding_dim=64)
sim_cluster.load_interactions_from_csv('interaction_data.csv')

sim_cluster.compute_embeddings()
sim_cluster.perform_faiss_clustering()

# Get recommended shorts for a user based on watched videos
watched_videos = ["content_1159", "content_3238"]
candidates = sim_cluster.get_video_candidates(watched_videos, top_n=20)
print("Recommended Shorts:", candidates)
