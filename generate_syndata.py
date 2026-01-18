#!/usr/bin/env python3
"""
Generate synthetic high-dimensional vectors for ANN evaluation
"""
import numpy as np
import json
import os
from pathlib import Path

# Configuration
NUM_VECTORS = 10000
VECTOR_DIM = 128
NUM_QUERIES = 100
SEED = 42

np.random.seed(SEED)

# Determine paths - data in exercises/ai-dos/data/
script_dir = Path(__file__).parent.absolute()
data_dir = script_dir / "data"
vectors_dir = data_dir / "syndata-vectors"
queries_dir = data_dir / "syndata-queries"

# Create directories
os.makedirs(vectors_dir, exist_ok=True)
os.makedirs(queries_dir, exist_ok=True)

print(f"Data will be saved to: {data_dir}")

print(f"Generating {NUM_VECTORS} vectors of dimension {VECTOR_DIM}...")

# Generate base vectors with some structure (clusters)
# Create 5 clusters with different centers
cluster_centers = np.random.randn(5, VECTOR_DIM)
cluster_centers = cluster_centers / np.linalg.norm(cluster_centers, axis=1, keepdims=True)

vectors = []
vector_ids = []

for i in range(NUM_VECTORS):
    # Assign to a cluster
    cluster_id = i % 5
    center = cluster_centers[cluster_id]
    
    # Generate vector around cluster center with some noise
    vector = center + np.random.randn(VECTOR_DIM) * 0.3
    # Normalize to unit length for cosine similarity
    vector = vector / np.linalg.norm(vector)
    
    vectors.append(vector.tolist())
    vector_ids.append(f"vec_{i:05d}")

# Save vectors as JSON
vectors_data = {
    "dimension": VECTOR_DIM,
    "count": NUM_VECTORS,
    "vectors": dict(zip(vector_ids, vectors))
}

with open(vectors_dir / 'vectors.json', 'w') as f:
    json.dump(vectors_data, f)

# Also save as numpy format for easier loading
vectors_array = np.array(vectors)
np.save(vectors_dir / 'vectors.npy', vectors_array)
np.save(vectors_dir / 'vector_ids.npy', np.array(vector_ids))

print(f"Saved vectors to {vectors_dir}/vectors.json and {vectors_dir}/vectors.npy")

# Generate query vectors
print(f"Generating {NUM_QUERIES} query vectors...")

queries = []
query_ids = []

for i in range(NUM_QUERIES):
    # Generate query vector (some near clusters, some random)
    if i < NUM_QUERIES // 2:
        # Queries near cluster centers
        cluster_id = i % 5
        center = cluster_centers[cluster_id]
        query = center + np.random.randn(VECTOR_DIM) * 0.2
    else:
        # Random queries
        query = np.random.randn(VECTOR_DIM)
    
    # Normalize
    query = query / np.linalg.norm(query)
    queries.append(query.tolist())
    query_ids.append(f"query_{i:03d}")

# Save queries
queries_data = {
    "dimension": VECTOR_DIM,
    "count": NUM_QUERIES,
    "queries": dict(zip(query_ids, queries))
}

with open(queries_dir / 'queries.json', 'w') as f:
    json.dump(queries_data, f)

queries_array = np.array(queries)
np.save(queries_dir / 'queries.npy', queries_array)
np.save(queries_dir / 'query_ids.npy', np.array(query_ids))

print(f"Saved queries to {queries_dir}/queries.json and {queries_dir}/queries.npy")

# Generate ground truth for first 10 queries (for evaluation)
print("Generating ground truth for first 10 queries...")

ground_truth = {}
vectors_np = np.array(vectors)

for i in range(min(10, NUM_QUERIES)):
    query = queries_array[i]
    # Compute cosine similarity with all vectors
    similarities = np.dot(vectors_np, query)
    # Get top 100
    top_indices = np.argsort(similarities)[::-1][:100]
    top_scores = similarities[top_indices]
    
    ground_truth[query_ids[i]] = {
        "indices": top_indices.tolist(),
        "scores": top_scores.tolist(),
        "vector_ids": [vector_ids[idx] for idx in top_indices]
    }

with open(queries_dir / 'ground_truth.json', 'w') as f:
    json.dump(ground_truth, f, indent=2)

print(f"Saved ground truth to {queries_dir}/ground_truth.json")

# Print statistics
print("\nDataset Statistics:")
print(f"  Vectors: {NUM_VECTORS}")
print(f"  Dimension: {VECTOR_DIM}")
print(f"  Queries: {NUM_QUERIES}")
print(f"  Ground truth queries: {len(ground_truth)}")
print(f"  Vector size: {vectors_array.nbytes / 1024 / 1024:.2f} MB")
