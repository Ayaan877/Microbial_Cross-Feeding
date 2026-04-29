from calculate_autoNet_yield import splitByDemand
from calculate_autoNet_yield_iterative import splitByDemandIterative
from load_data import *
import sys
import time
import multiprocessing as mp
import pickle
import os
from directory_paths import resolve_crossnet_path, resolve_merged_yield_path


def compute_yield_sbd(merged_net):
    return splitByDemand(
        stoich_matrix, rxnMat, prodMat,
        sumRxnVec, rho, pi, nutrientSet,
        Energy, Currency, Core, merged_net)


def compute_yield_iter(merged_net):
    return splitByDemandIterative(
        stoich_matrix, rxnMat, prodMat,
        sumRxnVec, rho, pi, nutrientSet,
        Energy, Currency, Core, merged_net)


def merge_pair(crossPair):
    """Return the union of both organisms' reaction indices."""
    cross_A = np.array(crossPair['cross_A'], dtype=int)
    cross_B = np.array(crossPair['cross_B'], dtype=int)
    return np.union1d(cross_A, cross_B)


if __name__ == "__main__":

    # Usage: get_merged_crossNet_yields.py <crossnet_subdir> <crossnet_file> <yield_mode> <num_workers>
    #
    #   crossnet_subdir : "crossnets_{source}_cv{cv}"   e.g. "crossnets_mp_cv1"
    #   crossnet_file   : "{byp|int}_{P|NP}"            e.g. "int_P"
    #   yield_mode      : sbd | iter
    #   num_workers     : parallel worker count

    crossnet_subdir = sys.argv[1]
    crossnet_file   = sys.argv[2]
    yield_mode      = sys.argv[3]
    num_workers     = int(sys.argv[4])

    if yield_mode == "sbd":
        compute_yield = compute_yield_sbd
    elif yield_mode == "iter":
        compute_yield = compute_yield_iter
    else:
        raise ValueError(f"Unknown yield_mode '{yield_mode}'. Use 'sbd' or 'iter'.")

    crossnet_path = resolve_crossnet_path(crossnet_subdir, crossnet_file)
    output_path   = resolve_merged_yield_path(crossnet_subdir, crossnet_file, yield_mode)
    os.makedirs(output_path.parent, exist_ok=True)

    with open(crossnet_path, "rb") as f:
        CrossNets = pickle.load(f)

    num_pairs = len(CrossNets)
    print(f"Loaded {num_pairs} cross-feeding pairs from {crossnet_path}")

    # Pre-compute all merged networks (union of both organisms' reactions).
    merged_nets = [merge_pair(p) for p in CrossNets]

    E_yields  = np.zeros(num_pairs)
    B_yields  = np.zeros(num_pairs)
    viability = np.zeros(num_pairs, dtype=bool)

    start = time.time()

    print(f"Using {num_workers} parallel workers ({yield_mode} mode)")
    with mp.Pool(processes=num_workers) as pool:
        for i, (E_yield, B_yield, status) in enumerate(
                pool.imap(compute_yield, merged_nets, chunksize=64)):
            E_yields[i]  = E_yield
            B_yields[i]  = B_yield
            viability[i] = status

            if (i + 1) % 500 == 0:
                processed_ratio = (i + 1) / num_pairs
                viable_ratio    = np.sum(viability[:i+1]) / (i + 1)
                print(f"  Processed {i + 1}/{num_pairs} ({processed_ratio:.2%}), "
                      f"viable: {np.sum(viability[:i+1])}/{i + 1} ({viable_ratio:.2%})")

    elapsed = time.time() - start
    valid = np.sum(viability)
    print(f"\nCompleted in {elapsed:.2f} seconds")
    print(f"Viable merged networks: {valid}/{num_pairs}")

    with open(output_path, "wb") as f:
        pickle.dump((E_yields, B_yields, viability), f)
    print(f"Saved merged yields to {output_path}")
