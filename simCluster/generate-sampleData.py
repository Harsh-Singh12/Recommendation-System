import pandas as pd
import numpy as np
import random
import time

# Generate Sample User Data
num_users = 1000
user_data = pd.DataFrame({
    'userid': [f'user_{i}' for i in range(num_users)],
    'age': np.random.randint(18, 60, num_users),
    'gender': np.random.choice(['Male', 'Female', 'Other'], num_users),
    'interests': np.random.choice(['Sports', 'Music', 'Tech', 'Fashion', 'Gaming'], num_users)
})
user_data.to_csv('user_metadata.csv', index=False)

# Generate Sample Content Data
num_contents = 5000
content_data = pd.DataFrame({
    'content_id': [f'content_{i}' for i in range(num_contents)],
    'title': [f'Video {i}' for i in range(num_contents)],
    'category': np.random.choice(['Sports', 'Music', 'Tech', 'Fashion', 'Gaming'], num_contents),
    'hashtags': [f'#tag{i % 10}' for i in range(num_contents)],
    'duration': np.random.randint(10, 300, num_contents),
    'upload_time': [time.time() - np.random.randint(0, 86400) for _ in range(num_contents)],
    'view_count': np.random.randint(0, 100000, num_contents)
})
content_data.to_csv('content_metadata.csv', index=False)

# Generate Sample Interaction Data
num_interactions = 20000
interaction_data = pd.DataFrame({
    'userid': np.random.choice(user_data['userid'], num_interactions),
    'content_id': np.random.choice(content_data['content_id'], num_interactions),
    'views': np.random.randint(1, 10, num_interactions),
    'likes': np.random.randint(0, 5, num_interactions),
    'shares': np.random.randint(0, 3, num_interactions),
    'comments': np.random.randint(0, 3, num_interactions),
    'watch_time': np.random.randint(5, 300, num_interactions),
    'engagement_score': np.random.rand(num_interactions)
})
interaction_data.to_csv('interaction_data.csv', index=False)

print("Sample data generated and saved to CSV files.")
