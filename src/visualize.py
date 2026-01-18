#!/usr/bin/env python3
"""Generate comparison graphs for HNSW vs Brute Force."""

import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import pickle
import json
import time
from pathlib import Path


def run_benchmark_with_details(index, queries, ground_truth, vectors, k=10):
    """Run benchmark and collect per-query data."""
    from .brute_force import brute_force_search
    from .benchmark import compute_recall
    
    bf_times = []
    hnsw_times = []
    bf_recalls = []
    hnsw_recalls = []
    
    for i in range(min(10, len(queries))):
        query = queries[i]
        query_id = f"query_{i:03d}"
        true_idx = ground_truth[query_id]["indices"][:k]
        
        # Brute force
        start = time.perf_counter()
        bf_res = brute_force_search(query, vectors, k)
        bf_times.append((time.perf_counter() - start) * 1000)
        bf_recalls.append(compute_recall([r[0] for r in bf_res], true_idx))
        
        # HNSW
        start = time.perf_counter()
        hnsw_res = index.query(query, k, ef_search=100)
        hnsw_times.append((time.perf_counter() - start) * 1000)
        hnsw_recalls.append(compute_recall([r[0] for r in hnsw_res], true_idx))
    
    return {
        "bf_times": bf_times,
        "hnsw_times": hnsw_times,
        "bf_recalls": bf_recalls,
        "hnsw_recalls": hnsw_recalls
    }


def plot_speed_comparison(data, save_path):
    """Plot speed comparison."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Bar chart per query
    ax = axes[0]
    x = np.arange(len(data["bf_times"]))
    width = 0.35
    bars1 = ax.bar(x - width/2, data["bf_times"], width, label='Brute Force', color='#E53935')
    bars2 = ax.bar(x + width/2, data["hnsw_times"], width, label='HNSW', color='#43A047')
    ax.set_xlabel('Query Index', fontsize=12)
    ax.set_ylabel('Latency (ms)', fontsize=12)
    ax.set_title('Query Latency Comparison', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.set_xticks(x)
    ax.grid(True, alpha=0.3, axis='y')
    
    # Summary bar
    ax = axes[1]
    methods = ['Brute Force', 'HNSW']
    times = [np.mean(data["bf_times"]), np.mean(data["hnsw_times"])]
    colors = ['#E53935', '#43A047']
    bars = ax.bar(methods, times, color=colors, width=0.5)
    ax.set_ylabel('Avg Latency (ms)', fontsize=12)
    ax.set_title('Average Query Time', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    
    for bar, t in zip(bars, times):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f'{t:.2f}ms', ha='center', fontsize=12, fontweight='bold')
    
    speedup = times[0] / times[1] if times[1] > 0 else 0
    ax.text(0.5, 0.95, f'Speedup: {speedup:.2f}x', transform=ax.transAxes,
            ha='center', fontsize=13, fontweight='bold',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


def plot_accuracy_comparison(data, save_path):
    """Plot accuracy (recall) comparison."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Recall per query
    ax = axes[0]
    x = np.arange(len(data["hnsw_recalls"]))
    ax.bar(x, data["hnsw_recalls"], color='#1976D2', alpha=0.8)
    ax.axhline(y=1.0, color='#43A047', linestyle='--', linewidth=2, label='Brute Force (100%)')
    ax.axhline(y=np.mean(data["hnsw_recalls"]), color='#E53935', linestyle='-', 
               linewidth=2, label=f'HNSW Avg ({np.mean(data["hnsw_recalls"]):.1%})')
    ax.set_xlabel('Query Index', fontsize=12)
    ax.set_ylabel('Recall@10', fontsize=12)
    ax.set_title('Recall per Query', fontsize=14, fontweight='bold')
    ax.set_ylim([0, 1.1])
    ax.set_xticks(x)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')
    
    # Summary
    ax = axes[1]
    methods = ['Brute Force', 'HNSW']
    recalls = [100, np.mean(data["hnsw_recalls"]) * 100]
    colors = ['#E53935', '#43A047']
    bars = ax.bar(methods, recalls, color=colors, width=0.5)
    ax.set_ylabel('Recall@10 (%)', fontsize=12)
    ax.set_title('Average Recall', fontsize=14, fontweight='bold')
    ax.set_ylim([0, 110])
    ax.grid(True, alpha=0.3, axis='y')
    
    for bar, r in zip(bars, recalls):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f'{r:.1f}%', ha='center', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


def plot_combined_summary(data, save_path):
    """Combined speed vs accuracy tradeoff."""
    fig, ax = plt.subplots(figsize=(8, 6))
    
    bf_time = np.mean(data["bf_times"])
    hnsw_time = np.mean(data["hnsw_times"])
    bf_recall = 100
    hnsw_recall = np.mean(data["hnsw_recalls"]) * 100
    
    ax.scatter([bf_time], [bf_recall], s=300, c='#E53935', label='Brute Force', zorder=5, edgecolor='black')
    ax.scatter([hnsw_time], [hnsw_recall], s=300, c='#43A047', label='HNSW', zorder=5, edgecolor='black')
    
    ax.annotate('Brute Force', (bf_time, bf_recall), xytext=(10, 10), 
                textcoords='offset points', fontsize=12, fontweight='bold')
    ax.annotate('HNSW', (hnsw_time, hnsw_recall), xytext=(10, -15),
                textcoords='offset points', fontsize=12, fontweight='bold')
    
    ax.set_xlabel('Latency (ms)', fontsize=12)
    ax.set_ylabel('Recall@10 (%)', fontsize=12)
    ax.set_title('Speed vs Accuracy Tradeoff', fontsize=14, fontweight='bold')
    ax.set_ylim([0, 110])
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=11)
    
    # Add annotation
    speedup = bf_time / hnsw_time if hnsw_time > 0 else 0
    ax.text(0.98, 0.02, f'HNSW: {speedup:.2f}x faster, {hnsw_recall:.1f}% recall',
            transform=ax.transAxes, ha='right', fontsize=11,
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9))
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


def main():
    """Generate all comparison graphs."""
    output_dir = Path("output")
    output_dir.mkdir(exist_ok=True)
    
    # Try fmnist first, fallback to syndata
    index_path = output_dir / "fmnist_index.pkl"
    queries_dir = Path("data/fmnist-queries")
    
    if not index_path.exists():
        index_path = output_dir / "index.pkl"
        queries_dir = Path("data/syndata-queries")
    
    print(f"Loading index: {index_path}")
    with open(index_path, "rb") as f:
        index = pickle.load(f)
    
    print(f"Loading queries: {queries_dir}")
    queries = np.load(queries_dir / "queries.npy")
    with open(queries_dir / "ground_truth.json") as f:
        ground_truth = json.load(f)
    
    print(f"\nRunning benchmark...")
    data = run_benchmark_with_details(index, queries, ground_truth, index.vectors)
    
    print(f"\nGenerating graphs...")
    plot_speed_comparison(data, str(output_dir / "speed_comparison.png"))
    plot_accuracy_comparison(data, str(output_dir / "accuracy_comparison.png"))
    plot_combined_summary(data, str(output_dir / "tradeoff_summary.png"))
    
    print(f"\n✓ All graphs saved to {output_dir}/")
    print(f"  - speed_comparison.png")
    print(f"  - accuracy_comparison.png")
    print(f"  - tradeoff_summary.png")


if __name__ == "__main__":
    main()
