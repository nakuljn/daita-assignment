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

| Dataset | HNSW Recall@10 | Speedup |
|---------|----------------|---------|
| Synthetic | 94% | ~1x |
| Fashion-MNIST | 96% | 1.19x |

**Note:** With 10,000 vectors, brute force using NumPy's vectorized operations is already very fast (~0.6-0.9ms). HNSW's advantage becomes more pronounced at larger scales (100k+ vectors) where O(log n) search significantly outperforms O(n) brute force.

## Graphs

Generated visualizations in `output/`:
- `speed_comparison.png` - Per-query latency comparison
- `accuracy_comparison.png` - Recall@10 per query
- `tradeoff_summary.png` - Speed vs accuracy tradeoff
