#!/usr/bin/env python3
"""
Generate Fashion-MNIST embeddings for ANN evaluation
Downloads Fashion-MNIST, generates embeddings, and creates query/ground truth sets
"""
import numpy as np
import json
import os
import urllib.request
import gzip
import struct
from pathlib import Path

# Configuration
NUM_VECTORS = 10000  # Use subset of Fashion-MNIST (full dataset has 60K)
VECTOR_DIM = 128
NUM_QUERIES = 100
SEED = 42

np.random.seed(SEED)

# Fashion-MNIST URLs
BASE_URL = "http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/"
TRAIN_IMAGES = "train-images-idx3-ubyte.gz"
TRAIN_LABELS = "train-labels-idx1-ubyte.gz"
TEST_IMAGES = "t10k-images-idx3-ubyte.gz"
TEST_LABELS = "t10k-labels-idx1-ubyte.gz"

def download_file(url, filename):
    """Download a file if it doesn't exist"""
    if os.path.exists(filename):
        print(f"  {filename} already exists, skipping download")
        return
    print(f"  Downloading {filename}...")
    urllib.request.urlretrieve(url, filename)
    print(f"  Downloaded {filename}")

def load_mnist_images(filename):
    """Load MNIST images from IDX file format"""
    with gzip.open(filename, 'rb') as f:
        magic, num, rows, cols = struct.unpack('>IIII', f.read(16))
        images = np.frombuffer(f.read(), dtype=np.uint8)
        images = images.reshape(num, rows * cols)
    return images.astype(np.float32)

def load_mnist_labels(filename):
    """Load MNIST labels from IDX file format"""
    with gzip.open(filename, 'rb') as f:
        magic, num = struct.unpack('>II', f.read(8))
        labels = np.frombuffer(f.read(), dtype=np.uint8)
    return labels

def simple_embedding(images, dim=128):
    """
    Generate embeddings from Fashion-MNIST images using a simple approach:
    1. Normalize pixel values
    2. Apply random projection for dimensionality reduction
    3. Normalize to unit vectors for cosine similarity
    """
    # Normalize pixel values to [0, 1]
    images_norm = images / 255.0
    
    # Simple embedding: Use random projection
    # For simplicity, we'll use random projection (can be replaced with actual PCA)
    # In practice, you might use a trained autoencoder or CNN
    np.random.seed(SEED)
    projection_matrix = np.random.randn(images_norm.shape[1], dim)
    # Normalize projection matrix columns
    col_norms = np.linalg.norm(projection_matrix, axis=0, keepdims=True)
    col_norms[col_norms == 0] = 1
    projection_matrix = projection_matrix / col_norms
    
    # Project to lower dimension
    embeddings = images_norm @ projection_matrix
    
    # Normalize to unit vectors for cosine similarity
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms[norms == 0] = 1  # Avoid division by zero
    embeddings = embeddings / norms
    
    return embeddings.astype(np.float32)

def create_ground_truth(queries, vectors, k=100):
    """Create ground truth by computing exact cosine similarity"""
    ground_truth = {}
    for i, query in enumerate(queries):
        # Compute cosine similarity (dot product for normalized vectors)
        similarities = np.dot(vectors, query)
        # Get top k
        top_indices = np.argsort(similarities)[::-1][:k]
        top_scores = similarities[top_indices]
        
        query_id = f"query_{i:03d}"
        ground_truth[query_id] = {
            "indices": top_indices.tolist(),
            "scores": top_scores.tolist(),
            "vector_ids": [f"vec_{idx:05d}" for idx in top_indices]
        }
    return ground_truth

# Determine paths - data in exercises/ai-dos/data/
script_dir = Path(__file__).parent.absolute()
data_dir = script_dir / "data"
vectors_dir = data_dir / "fmnist-vectors"
queries_dir = data_dir / "fmnist-queries"
raw_data_dir = data_dir / "fmnist-raw"

# Create directories
os.makedirs(vectors_dir, exist_ok=True)
os.makedirs(queries_dir, exist_ok=True)
os.makedirs(raw_data_dir, exist_ok=True)

print(f"Data will be saved to: {data_dir}")

print("=" * 60)
print("Fashion-MNIST Dataset Generator")
print("=" * 60)

# Download Fashion-MNIST dataset
print("\n1. Downloading Fashion-MNIST dataset...")
download_file(BASE_URL + TRAIN_IMAGES, str(raw_data_dir / TRAIN_IMAGES))
download_file(BASE_URL + TRAIN_LABELS, str(raw_data_dir / TRAIN_LABELS))
download_file(BASE_URL + TEST_IMAGES, str(raw_data_dir / TEST_IMAGES))
download_file(BASE_URL + TEST_LABELS, str(raw_data_dir / TEST_LABELS))

# Load data
print("\n2. Loading Fashion-MNIST data...")
train_images = load_mnist_images(str(raw_data_dir / TRAIN_IMAGES))
train_labels = load_mnist_labels(str(raw_data_dir / TRAIN_LABELS))
test_images = load_mnist_images(str(raw_data_dir / TEST_IMAGES))
test_labels = load_mnist_labels(str(raw_data_dir / TEST_LABELS))

print(f"  Training images: {train_images.shape}")
print(f"  Test images: {test_images.shape}")

# Combine train and test, then sample
all_images = np.vstack([train_images, test_images])
all_labels = np.concatenate([train_labels, test_labels])

# Sample NUM_VECTORS images (stratified by class if possible)
print(f"\n3. Sampling {NUM_VECTORS} vectors...")
if NUM_VECTORS <= len(all_images):
    # Random sample
    indices = np.random.choice(len(all_images), NUM_VECTORS, replace=False)
    sampled_images = all_images[indices]
    sampled_labels = all_labels[indices]
else:
    sampled_images = all_images
    sampled_labels = all_labels

print(f"  Sampled {len(sampled_images)} images")
print(f"  Label distribution: {np.bincount(sampled_labels)}")

# Generate embeddings
print(f"\n4. Generating {VECTOR_DIM}-dimensional embeddings...")
embeddings = simple_embedding(sampled_images, dim=VECTOR_DIM)
print(f"  Embeddings shape: {embeddings.shape}")

# Save vectors
print("\n5. Saving vectors...")
vector_ids = [f"vec_{i:05d}" for i in range(len(embeddings))]
vectors_dict = {vid: vec.tolist() for vid, vec in zip(vector_ids, embeddings)}

vectors_data = {
    "dimension": VECTOR_DIM,
    "count": len(embeddings),
    "source": "Fashion-MNIST",
    "labels": sampled_labels.tolist(),
    "label_names": [
        "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
        "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"
    ],
    "vectors": vectors_dict
}

with open(vectors_dir / 'vectors.json', 'w') as f:
    json.dump(vectors_data, f)

np.save(vectors_dir / 'vectors.npy', embeddings)
np.save(vectors_dir / 'vector_ids.npy', np.array(vector_ids))
np.save(vectors_dir / 'labels.npy', sampled_labels)

print(f"  Saved to {vectors_dir}/vectors.json and {vectors_dir}/vectors.npy")

# Generate query vectors from test set
print(f"\n6. Generating {NUM_QUERIES} query vectors...")
# Use test images for queries
query_indices = np.random.choice(len(test_images), min(NUM_QUERIES, len(test_images)), replace=False)
query_images = test_images[query_indices]
query_labels = test_labels[query_indices]

query_embeddings = simple_embedding(query_images, dim=VECTOR_DIM)
query_ids = [f"query_{i:03d}" for i in range(len(query_embeddings))]

queries_dict = {qid: qvec.tolist() for qid, qvec in zip(query_ids, query_embeddings)}

queries_data = {
    "dimension": VECTOR_DIM,
    "count": len(query_embeddings),
    "source": "Fashion-MNIST (test set)",
    "labels": query_labels.tolist(),
    "label_names": [
        "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
        "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"
    ],
    "queries": queries_dict
}

with open(queries_dir / 'queries.json', 'w') as f:
    json.dump(queries_data, f)

np.save(queries_dir / 'queries.npy', query_embeddings)
np.save(queries_dir / 'query_ids.npy', np.array(query_ids))
np.save(queries_dir / 'query_labels.npy', query_labels)

print(f"  Saved to {queries_dir}/queries.json and {queries_dir}/queries.npy")

# Generate ground truth for first 10 queries
print("\n7. Computing ground truth for first 10 queries...")
ground_truth = create_ground_truth(
    query_embeddings[:10],
    embeddings,
    k=100
)

with open(queries_dir / 'ground_truth.json', 'w') as f:
    json.dump(ground_truth, f, indent=2)

print(f"  Saved ground truth to {queries_dir}/ground_truth.json")

# Print statistics
print("\n" + "=" * 60)
print("Dataset Statistics:")
print("=" * 60)
print(f"  Vectors: {len(embeddings):,}")
print(f"  Dimension: {VECTOR_DIM}")
print(f"  Queries: {len(query_embeddings)}")
print(f"  Ground truth queries: {len(ground_truth)}")
print(f"  Vector size: {embeddings.nbytes / 1024 / 1024:.2f} MB")
print(f"  Label distribution: {dict(zip(*np.unique(sampled_labels, return_counts=True)))}")
print("=" * 60)
print(f"\nDataset ready! Use {vectors_dir}/ and {queries_dir}/ directories.")
