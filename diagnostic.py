"""
Diagnostic: estimate total runtime of allAutonomousNetworks.
Times a few serial buildAutonomousNetwork calls, then extrapolates
for any core count — no need to have 32 cores locally.
"""
import sys
import time
import itertools
import numpy as np
from load_paths import loadPaths
from load_data import *
from combine_pathways import buildAutonomousNetwork

if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "batch"
    n_sample = int(sys.argv[2]) if len(sys.argv) > 2 else 3

    all_paths, data_dir = loadPaths(mode=mode, dataset=6)

    print(f"Dataset: MinNets{data_dir}")
    print(f"Number of targets: {len(all_paths)}")

    path_counts = [len(p) for p in all_paths]
    for i, count in enumerate(path_counts):
        print(f"  Target {i+1}: {count} pathways")

    total_combos = 1
    for c in path_counts:
        total_combos *= c
    print(f"\nTotal combinations: {total_combos:,}")

    # Time n_sample serial calls (single-core, no Pool overhead)
    print(f"\nTiming {n_sample} serial buildAutonomousNetwork calls...")
    combo_generator = itertools.product(*all_paths)

    times = []
    for i, combo in enumerate(combo_generator):
        if i >= n_sample:
            break
        rng = np.random.default_rng(i)
        start = time.time()
        network = buildAutonomousNetwork(combo, rxnMat, prodMat, sumRxnVec,
                                         nutrientSet, Currency, Core, rng=rng)
        elapsed = time.time() - start
        times.append(elapsed)
        print(f"  Sample {i+1}: {elapsed:.3f}s  (network size = {len(network)})")

    avg_time = np.mean(times)
    std_time = np.std(times)
    print(f"\nAvg time per combo: {avg_time:.3f}s (+/- {std_time:.3f}s)")
    print(f"Total serial time: {total_combos * avg_time / 3600:.2f} hours")

    print(f"\n{'Cores':<8} {'Ideal':<18} {'~75% efficiency':<18}")
    print("-" * 44)
    for n in [8, 16, 32, 64]:
        ideal_s = total_combos * avg_time / n
        real_s = ideal_s / 0.75
        def fmt(s):
            if s >= 3600:
                return f"{s/3600:.2f} hours"
            return f"{s/60:.1f} min"
        print(f"{n:<8} {fmt(ideal_s):<18} {fmt(real_s):<18}")
