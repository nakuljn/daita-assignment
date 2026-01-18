# Solution Report

## Implementation

### Core Components
- **Cosine Similarity** (`src/similarity.py`) - Vectorized using numpy dot product
- **HNSW Index** (`src/hnsw_index.py`) - Full HNSW algorithm implementation
- **Brute Force** (`src/brute_force.py`) - Exact search baseline
- **Benchmarking** (`src/benchmark.py`) - Recall and latency measurement

### HNSW Algorithm
- Multi-layer navigable small world graph
- Probabilistic level assignment: `level = -ln(random) / ln(M)`
- Greedy search at upper layers, beam search at layer 0
- Parameters: M=16, ef_construction=100

## Usage

```bash
# Generate data
python generate_syndata.py

# Run demo with benchmark
python -m src.main

# Or build and query separately
python -m src.build
python -m src.query
```

## Results

On synthetic dataset (10,000 vectors, 128-dim):
- HNSW achieves ~90-100% Recall@10
- 10-50x speedup vs brute force
