"""
Benchmark single sequential pruning calls per core metabolite, then estimate
parallel wall time for N_WORKERS=8 workers (as used on the cluster).

Wall-time model per attempt:
  - N_WORKERS independent calls run in parallel; bottleneck = slowest worker.
  - E[max of k i.i.d. samples] ≈ mean + std * sqrt(2 * ln(k))  (Gumbel approx)
  - Lower bound  = mean single call time
  - Upper bound  = above Gumbel estimate for k = N_WORKERS

Total projected runtime = wall_time_per_attempt × MAX_ATTEMPTS (no-plateau worst case).
"""
import math
import time
import numpy as np
from load_data import rxnMat, prodMat, sumRxnVec, nutrientSet, Currency, met_map, inv_met_map
from reverse_scope import giveRevScope
from batch_pruning import randMinNetwork

N_WORKERS   = 8     # workers on cluster (for projection only — no Pool used here)
N_REPS      = 10    # sequential single-call repetitions per target
MAX_ATTEMPTS = 500  # worst-case attempts if plateau never triggers

KEGG_CORE_IDS = ["C00022", "C00041", "C00097", "C00025",
                 "C00065", "C00117", "C00013", "C00009"]


def gumbel_expected_max(mean, std, k):
    """E[max of k i.i.d. normal samples] via Gumbel approximation."""
    if std == 0 or k <= 1:
        return mean
    return mean + std * math.sqrt(2 * math.log(k))


if __name__ == "__main__":
    print(f"\nBenchmarking batch pruning: {N_REPS} sequential calls per target")
    print(f"Cluster projection: N_WORKERS={N_WORKERS}, MAX_ATTEMPTS={MAX_ATTEMPTS}\n")

    hdr = (f"{'KEGG ID':<10} {'Target':>8}  "
           f"{'mean (s)':>10} {'std (s)':>9} {'max (s)':>9}  "
           f"{'Est. wall/att (s)':>18}  "
           f"{'Proj lo (hrs)':>14} {'Proj hi (hrs)':>14}")
    print(hdr)
    print("-" * len(hdr))

    all_results = {}

    for kegg_id in KEGG_CORE_IDS:
        target    = met_map[kegg_id]
        target_id = inv_met_map[target]

        satMets, satRxns = giveRevScope(rxnMat, prodMat, sumRxnVec,
                                        nutrientSet, Currency, target)

        call_times = []
        for i in range(N_REPS):
            rng = np.random.default_rng()
            t0  = time.perf_counter()
            randMinNetwork(satRxns, rxnMat, prodMat, sumRxnVec,
                           target, nutrientSet, Currency, rng=rng)
            call_times.append(time.perf_counter() - t0)

        mean_s = np.mean(call_times)
        std_s  = np.std(call_times, ddof=1) if N_REPS > 1 else 0.0
        max_s  = np.max(call_times)

        # Wall time per attempt: lower = mean (if all workers finish together),
        # upper = expected max over N_WORKERS i.i.d. draws (Gumbel)
        wall_lo = mean_s
        wall_hi = gumbel_expected_max(mean_s, std_s, N_WORKERS)

        proj_lo_hrs = (wall_lo * MAX_ATTEMPTS) / 3600.0
        proj_hi_hrs = (wall_hi * MAX_ATTEMPTS) / 3600.0

        all_results[kegg_id] = dict(
            target_id=target_id, mean_s=mean_s, std_s=std_s, max_s=max_s,
            wall_lo=wall_lo, wall_hi=wall_hi,
            proj_lo_hrs=proj_lo_hrs, proj_hi_hrs=proj_hi_hrs,
            call_times=call_times,
        )

        print(f"{target_id:<10} {kegg_id:>8}  "
              f"{mean_s:>10.2f} {std_s:>9.2f} {max_s:>9.2f}  "
              f"{wall_hi:>18.2f}  "
              f"{proj_lo_hrs:>14.2f} {proj_hi_hrs:>14.2f}",
              flush=True)

    print()
    worst = max(all_results.items(), key=lambda x: x[1]["proj_hi_hrs"])
    w = worst[1]
    print(f"Worst-case (no plateau): {w['target_id']} ({worst[0]})")
    print(f"  → {w['proj_lo_hrs']:.2f} – {w['proj_hi_hrs']:.2f} hrs over {MAX_ATTEMPTS} attempts with {N_WORKERS} workers")
    print()
    print("Individual call times per target (s):")
    for kegg_id, r in all_results.items():
        times_str = ", ".join(f"{t:.2f}" for t in r["call_times"])
        print(f"  {r['target_id']} ({kegg_id}): [{times_str}]")
