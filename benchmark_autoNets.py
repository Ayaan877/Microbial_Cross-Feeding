"""
Benchmark buildAutonomousNetwork for prune=True and prune=False.

Each call = one random combo of 8 pathways (one per core metabolite) → union → optional prune.
Serial calls only; parallel wall time projected using Gumbel expected-max over N_WORKERS.

Wall-time model per task:
  E[max of k i.i.d.] ≈ mean + std * sqrt(2 * ln(k))   (Gumbel approx, upper bound)
  lower bound         = mean

Total projected = wall_time_per_task × N_TARGET  (no-deduplication worst case)
In practice deduplication means fewer tasks are needed, so these are conservative.
"""
import math
import time
import numpy as np
from load_data import rxnMat, prodMat, sumRxnVec, nutrientSet, Currency, Core
from load_networks import load_minpaths
from combine_pathways import buildAutonomousNetwork

N_WORKERS     = 32     # as specified in run_autonomous_networks.pbs
N_REPS        = 10     # serial calls per mode
N_TARGET      = 50000  # target unique networks (from PBS config)
PATHS_VERSION = "2"


def gumbel_expected_max(mean, std, k):
    if std == 0 or k <= 1:
        return mean
    return mean + std * math.sqrt(2 * math.log(k))


def run_benchmark(all_paths, prune, n_reps, rng):
    path_counts = [len(p) for p in all_paths]
    times = []
    for _ in range(n_reps):
        combo = tuple(all_paths[t][rng.integers(path_counts[t])] for t in range(8))
        t0 = time.perf_counter()
        buildAutonomousNetwork(combo, rxnMat, prodMat, sumRxnVec,
                               nutrientSet, Currency, Core,
                               prune=prune, rng=np.random.default_rng())
        times.append(time.perf_counter() - t0)
    return times


if __name__ == "__main__":
    print(f"Loading minpaths from paths_pv{PATHS_VERSION}...")
    all_paths = load_minpaths(f"paths_pv{PATHS_VERSION}")
    path_counts = [len(p) for p in all_paths]
    print(f"Pathway counts per target: {path_counts}\n")

    rng = np.random.default_rng(42)

    hdr = (f"{'Mode':<10}  {'mean (s)':>10} {'std (s)':>9} {'min (s)':>9} {'max (s)':>9}  "
           f"{'Est. wall/batch (s)':>21}  "
           f"{'Proj lo (hrs)':>14} {'Proj hi (hrs)':>14}")
    print(hdr)
    print("-" * len(hdr))

    results = {}
    for prune, label in [(False, "NoPrune"), (True, "Prune")]:
        times = run_benchmark(all_paths, prune=prune, n_reps=N_REPS, rng=rng)

        mean_s = np.mean(times)
        std_s  = np.std(times, ddof=1)
        min_s  = np.min(times)
        max_s  = np.max(times)

        wall_lo = mean_s
        wall_hi = gumbel_expected_max(mean_s, std_s, N_WORKERS)

        # imap_unordered streams N_TARGET tasks across N_WORKERS in parallel.
        # Effective batches = N_TARGET / N_WORKERS; each batch takes wall_hi seconds.
        # (Ignoring duplicates — real runtime will be less than this worst case.)
        n_batches = N_TARGET / N_WORKERS
        proj_lo_hrs = (wall_lo * n_batches) / 3600.0
        proj_hi_hrs = (wall_hi * n_batches) / 3600.0

        results[label] = dict(mean_s=mean_s, std_s=std_s, min_s=min_s, max_s=max_s,
                              wall_lo=wall_lo, wall_hi=wall_hi,
                              proj_lo_hrs=proj_lo_hrs, proj_hi_hrs=proj_hi_hrs,
                              times=times)

        print(f"{label:<10}  {mean_s:>10.3f} {std_s:>9.3f} {min_s:>9.3f} {max_s:>9.3f}  "
              f"{wall_hi:>20.3f}  "
              f"{proj_lo_hrs:>14.2f} {proj_hi_hrs:>14.2f}",
              flush=True)

    print()
    print("Notes:")
    print(f"  Projection: {N_TARGET} tasks / {N_WORKERS} workers = {N_TARGET//N_WORKERS} parallel batches, no early duplicate termination.")
    print(f"  Real runtime < projection since duplicate combos reduce effective task count.")
    print()
    print("Individual call times (s):")
    for label, r in results.items():
        times_str = ", ".join(f"{t:.3f}" for t in r["times"])
        print(f"  {label}: [{times_str}]")
