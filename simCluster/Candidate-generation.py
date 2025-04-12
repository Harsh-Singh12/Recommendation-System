import pandas as pd
import torch
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv
from torch_geometric.data import Data
from sklearn.mixture import GaussianMixture
import os
from flask import Flask, request, jsonify
import time
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import schedule
import threading
from collections import Counter

app = Flask(__name__)

# Load Data
INTERACTION_FILE = 'interaction_data.csv'
CONTENT_FILE = 'content_metadata.csv'
USER_FILE = 'user_metadata.csv'

interaction_data = pd.read_csv(INTERACTION_FILE)
content_data = pd.read_csv(CONTENT_FILE)
user_data = pd.read_csv(USER_FILE)

# =====================================
# Pipeline 1: SimCluster Candidate Generation
# =====================================

def generate_simcluster_candidates():
    # SimCluster Code (Already Provided)
    # Returns: {user_id: [content_id1, content_id2, ...]}
    pass

# =====================================
# Pipeline 2: Content-Based Filtering
# =====================================

def generate_content_based_candidates():
    # Content-Based Filtering Code
    # Returns: {user_id: [content_id1, content_id2, ...]}
    pass

# =====================================
# Pipeline 3: Trending Content Generation
# =====================================

def generate_trending_candidates():
    # Trending Content Code
    # Returns: {user_id: [content_id1, content_id2, ...]}
    pass

# =====================================
# Pipeline 4: Popularity-Based Candidate Generation
# =====================================

def generate_popularity_based_candidates():
    # Popularity-Based Content Code
    # Returns: {user_id: [content_id1, content_id2, ...]}
    pass

# =====================================
# Merge All Candidates
# =====================================

def merge_all_candidates():
    simcluster_candidates = generate_simcluster_candidates()
    content_candidates = generate_content_based_candidates()
    trending_candidates = generate_trending_candidates()
    popularity_candidates = generate_popularity_based_candidates()

    candidate_list = {}
    for user_id in user_data['userid'].unique():
        combined_candidates = (
                set(simcluster_candidates.get(user_id, [])) |
                set(content_candidates.get(user_id, [])) |
                set(trending_candidates.get(user_id, [])) |
                set(popularity_candidates.get(user_id, []))
        )
        candidate_list[user_id] = list(combined_candidates)[:10000]

    pd.DataFrame({
        'userid': list(candidate_list.keys()),
        'candidates': list(candidate_list.values())
    }).to_csv('candidates.csv', index=False)

# =====================================
# Candidate Re-Ranking
# =====================================

def rerank_candidates():
    df = pd.read_csv('candidates.csv')

    def calculate_personal_score(user_id, content_id):
        watched_content = interaction_data[interaction_data['userid'] == user_id]['content_id'].tolist()
        if content_id in watched_content:
            return 0
        user_profile = user_data[user_data['userid'] == user_id].iloc[:, 1:].values
        content_profile = content_data[content_data['content_id'] == content_id].iloc[:, 1:].values
        similarity = cosine_similarity(user_profile, content_profile)[0][0]
        return similarity

    def apply_decay(content_id):
        content_age = (pd.Timestamp.now() - pd.to_datetime(content_data[content_data['content_id'] == content_id]['upload_time'])).days
        if content_age <= 1:
            return 1.5
        elif content_age <= 7:
            return 1.2
        else:
            return 1.0

    def calculate_diversity_boost(user_id, content_id):
        watched_tags = interaction_data[interaction_data['userid'] == user_id]['groupTag'].tolist()
        content_tag = content_data[content_data['content_id'] == content_id]['groupTag'].values[0]
        if content_tag in watched_tags:
            return 0.9
        else:
            return 1.2

    final_recommendations = []

    for index, row in df.iterrows():
        user_id = row['userid']
        candidates = row['candidates'].strip("[]").replace("'", "").split(', ')
        scores = []

        for content_id in candidates:
            personal_score = calculate_personal_score(user_id, content_id)
            decay_factor = apply_decay(content_id)
            diversity_boost = calculate_diversity_boost(user_id, content_id)
            final_score = personal_score * decay_factor * diversity_boost
            scores.append((content_id, final_score))

        scores.sort(key=lambda x: x[1], reverse=True)
        top_candidates = [x[0] for x in scores[:10000]]
        final_recommendations.append((user_id, top_candidates))

    pd.DataFrame(final_recommendations, columns=['userid', 'candidates']).to_csv('final_candidates.csv', index=False)

# =====================================
# Scheduling 12-Hour Batch Pipeline
# =====================================

schedule.every(12).hours.do(merge_all_candidates)
schedule.every(12).hours.do(rerank_candidates)

thread = threading.Thread(target=schedule.run_pending)
thread.start()

@app.route('/recommend', methods=['GET'])
def recommend():
    user_id = request.args.get('user_id')
    df = pd.read_csv('final_candidates.csv')

    if user_id not in df['userid'].values:
        return jsonify({"error": "User ID not found"}), 404

    candidates = df[df['userid'] == user_id]['candidates'].values[0]
    return jsonify({
        "user_id": user_id,
        "candidates": eval(candidates)
    })

if __name__ == '__main__':
    app.run(port=5000)

print("✅ Multi-Approach Candidate Generation with Re-ranking running every 12 hours.")
