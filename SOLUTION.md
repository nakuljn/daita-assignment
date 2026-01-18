# Vector Similarity Search Engine - Solution

## Overview

This project implements an **Approximate Nearest Neighbor (ANN)** search system using the **HNSW (Hierarchical Navigable Small World)** algorithm. HNSW is a state-of-the-art ANN algorithm that provides excellent recall while being significantly faster than brute-force search.

## Project Structure

```
├── src/                      # Source code
│   ├── similarity.py         # Cosine similarity functions
│   ├── brute_force.py        # Exact search baseline
│   ├── hnsw_index.py         # HNSW ANN implementation
│   ├── data_loader.py        # Data loading utilities
│   ├── benchmark.py          # Benchmarking utilities
│   ├── visualize.py          # Visualization tools
│   ├── build.py              # CLI: Build index
│   ├── query.py              # CLI: Query index
│   └── main.py               # Demo script
├── tests/                    # Unit tests
│   ├── test_similarity.py
│   ├── test_brute_force.py
│   └── test_hnsw_index.py
├── output/                   # Generated files
│   ├── index.pkl             # Synthetic data index
│   ├── fmnist_index.pkl      # Fashion-MNIST index
│   └── *.png                 # Benchmark graphs
├── data/                     # Data files (generated)
├── generate_syndata.py       # Synthetic data generator
├── generate_fmnist.py        # Fashion-MNIST generator
└── requirements.txt
```

## Quick Start

```bash
# 1. Setup
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# 2. Generate data
python3 generate_syndata.py

# 3. Build index
python -m src.build

# 4. Query
python -m src.query                  # Benchmark mode
python -m src.query --query 0 1 2    # Specific queries

# 5. Visualize
python -m src.visualize
```

## Implementation Details

### 1. Cosine Similarity (`src/similarity.py`)
- Efficient dot-product based similarity (vectors are pre-normalized)
- Batch computation for vectorized operations

### 2. HNSW Index (`src/hnsw_index.py`)
- **Build Phase**: O(n log n) - Insert vectors into multi-layer graph
- **Query Phase**: O(log n) - Navigate layers + beam search
- Parameters: M=16 (connections), ef_construction=100 (build beam width)

### 3. Brute Force (`src/brute_force.py`)
- O(n) exact search for baseline comparison
- Uses NumPy vectorization for efficiency

## Benchmark Results

### Synthetic Dataset (10,000 vectors, 128 dimensions)

| Method | Avg Latency | Recall@10 |
|--------|-------------|-----------|
| Brute Force | 0.97 ms | 100% |
| HNSW | 2.03 ms | 94% |

### Fashion-MNIST Dataset (10,000 vectors, 128 dimensions)

| Method | Avg Latency | Recall@10 |
|--------|-------------|-----------|
| Brute Force | 0.85 ms | 100% |
| HNSW | 0.84 ms | 96% |

**Key Insight**: HNSW achieves **96% recall** with comparable latency on Fashion-MNIST. The speedup advantage becomes more pronounced at larger scales (100k+ vectors).

## Test Results

```
tests/test_brute_force.py::test_returns_k_results PASSED
tests/test_brute_force.py::test_most_similar_is_self PASSED
tests/test_brute_force.py::test_results_sorted_descending PASSED
tests/test_hnsw_index.py::test_build_creates_layers PASSED
tests/test_hnsw_index.py::test_query_returns_k_results PASSED
tests/test_hnsw_index.py::test_self_is_top_result PASSED
tests/test_hnsw_index.py::test_recall_above_threshold PASSED
tests/test_similarity.py::test_identical_vectors PASSED
tests/test_similarity.py::test_orthogonal_vectors PASSED
tests/test_similarity.py::test_opposite_vectors PASSED
tests/test_similarity.py::test_batch_similarity PASSED

============================== 11 passed in 0.71s ==============================
```

## Performance Tradeoffs

### Speed vs Accuracy
- **Higher `ef_search`** → Better recall, slower queries
- **Higher `M`** → Better recall, more memory, slower build
- **More layers** → Faster navigation, more memory

### When to Use HNSW vs Brute Force
| Scenario | Recommendation |
|----------|----------------|
| < 10,000 vectors | Brute force (simpler, 100% accurate) |
| > 10,000 vectors | HNSW (faster at scale) |
| Need 100% recall | Brute force |
| Can tolerate 95%+ recall | HNSW |

### Build Time vs Query Time
- HNSW has expensive build (~14-23s for 10k vectors)
- But queries are fast and can be run many times
- Build once, query many times

## Generated Visualizations

The `python -m src.visualize` command generates:
- `output/speed_comparison.png` - Latency comparison
- `output/accuracy_comparison.png` - Recall comparison
- `output/tradeoff_summary.png` - Speed vs accuracy plot

## Algorithm Choice: Why HNSW?

| Algorithm | Pros | Cons |
|-----------|------|------|
| **HNSW** | Best recall, fast queries | Complex to implement |
| LSH | Simple | Lower recall |
| IVF | Good balance | Needs clustering |
| Annoy | Easy to use | Library (not allowed) |

HNSW was chosen for its excellent recall-latency tradeoff and because it can be implemented without external libraries.

