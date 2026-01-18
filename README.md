# Vector Similarity Search Engine

```
Time: 75 Minutes
```

## Problem

Implement an approximate nearest neighbor (ANN) system for high-dimensional vector similarity search.

## Dataset

Two datasets are available for evaluation:

### Synthetic Dataset (Default)
Located in `data/syndata-vectors/` and `data/syndata-queries/`:
- **10,000 vectors** of dimension 128 (normalized for cosine similarity)
- **100 query vectors** for testing
- **Ground truth** for first 10 queries (top-100 exact results) for evaluation

To generate, run:
```bash
python3 generate_syndata.py
```

### Fashion-MNIST Dataset (Optional)
Located in `data/fmnist-vectors/` and `data/fmnist-queries/`:
- **10,000 vectors** from Fashion-MNIST images (128-dimensional embeddings)
- **100 query vectors** from test set
- **Ground truth** for first 10 queries
- Includes class labels for additional evaluation metrics

To generate, run:
```bash
python3 generate_fmnist.py
```

Vectors are provided in both JSON and NumPy formats.

## Requirements

Your system should implement:

1. **Cosine Similarity**
   - Implement cosine similarity calculation
   - Support high-dimensional vectors

2. **Index Construction**
   - Build an efficient index structure for fast retrieval

3. **Query Top-K**
   - Implement query interface to find top-k nearest neighbors
   - Return results with similarity scores

4. **Comparison & Benchmarking**
   - Implement brute-force exact search for comparison
   - Compare ANN vs brute-force on accuracy and latency
   - Document performance tradeoffs

## Constraints

- Implement core algorithms yourself (don't use full ANN libraries like FAISS, Annoy, etc.)
- You may use basic data structures and math libraries

## Submission

- Create a public git repository containing your submission and share the repository link
- Do not fork this repository or create pull requests
