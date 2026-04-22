from calculate_crossfeeding_yield import splitByDemand_crossfeeding
from load_data import *
import time
import multiprocessing as mp
import pickle
import sys

def compute_crossfeeding_yield(crossPair):
    result = splitByDemand_crossfeeding(
        stoich_matrix, rxnMat, prodMat,
        sumRxnVec, rho, pi, nutrientSet,
        Energy, Currency, Core, crossPair)
    return (
        result['E_A'], result['B_A'], result['viable_A'],
        result['E_B'], result['B_B'], result['viable_B'],
        result['pair_viable'],
        result['flux_A_to_B'], result['flux_B_to_A'],
    )

if __name__ == "__main__":

    version = sys.argv[1]       # dataset version, e.g. 2
    crossnet_type = sys.argv[2] # byp or int

    if crossnet_type not in ("byp", "int"):
        print(f"Unknown crossnet_type '{crossnet_type}'. Use 'byp' or 'int'.")
        sys.exit(1)

    crossnet_dir = "CrossNets_revScope"
    crossnet_data = f"CrossNets_revScope{version}_{crossnet_type}{version}.pkl"
    crossnet_path = f"{crossnet_dir}/{crossnet_data}"
    num_workers = 32

    with open(crossnet_path, "rb") as f:
        CrossNets = pickle.load(f)

    num_pairs = len(CrossNets)
    print(f"Loaded {num_pairs} cross-feeding pairs from {crossnet_path}")

    E_A_yields   = np.zeros(num_pairs)
    B_A_yields   = np.zeros(num_pairs)
    viable_A     = np.zeros(num_pairs, dtype=bool)
    E_B_yields   = np.zeros(num_pairs)
    B_B_yields   = np.zeros(num_pairs)
    viable_B     = np.zeros(num_pairs, dtype=bool)
    pair_viable  = np.zeros(num_pairs, dtype=bool)
    flux_A_to_B  = np.zeros(num_pairs)
    flux_B_to_A  = np.zeros(num_pairs)

    start = time.time()

    print(f"Using {num_workers} parallel workers")
    with mp.Pool(processes=num_workers) as pool:
        for i, (EA, BA, vA, EB, BB, vB, vPair, fAB, fBA) in enumerate(
                pool.imap(compute_crossfeeding_yield, CrossNets, chunksize=64)):
            E_A_yields[i]  = EA
            B_A_yields[i]  = BA
            viable_A[i]    = vA
            E_B_yields[i]  = EB
            B_B_yields[i]  = BB
            viable_B[i]    = vB
            pair_viable[i] = vPair
            flux_A_to_B[i] = fAB
            flux_B_to_A[i] = fBA

            if (i + 1) % 500 == 0:
                processed_ratio = (i + 1) / num_pairs
                viable_ratio = np.sum(pair_viable[:i+1]) / (i + 1)
                print(f"  Processed {i + 1}/{num_pairs} ({processed_ratio:.2%}), "
                      f"pair viable: {np.sum(pair_viable[:i+1])}/{i + 1} ({viable_ratio:.2%})")

    elapsed = time.time() - start
    valid = np.sum(pair_viable)
    print(f"\nCompleted in {elapsed:.2f} seconds")
    print(f"Viable pairs (both organisms produce all precursors): {valid}/{num_pairs}")

    output_path = f"{crossnet_dir}/Yields_{crossnet_data}"
    with open(output_path, "wb") as f:
        pickle.dump({
            'E_A':        E_A_yields,
            'B_A':        B_A_yields,
            'viable_A':   viable_A,
            'E_B':        E_B_yields,
            'B_B':        B_B_yields,
            'viable_B':   viable_B,
            'pair_viable': pair_viable,
            'flux_A_to_B': flux_A_to_B,
            'flux_B_to_A': flux_B_to_A,
        }, f)
    print(f"Saved yields to {output_path}")
