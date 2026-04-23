from calculate_yield import *
from calculate_yield_iterative import *
from load_data import *
import sys
import time
import multiprocessing as mp
import pickle

def compute_yield_sbd(net):
    return splitByDemand(
        stoich_matrix, rxnMat, prodMat,
        sumRxnVec, rho, pi, nutrientSet,
        Energy, Currency, Core, net)

def compute_yield_iter(net):
    return splitByDemandIterative(
        stoich_matrix, rxnMat, prodMat,
        sumRxnVec, rho, pi, nutrientSet,
        Energy, Currency, Core, net)

if __name__ == "__main__":

    # Args supplied by PBS script (see run_autonomous_yields.pbs)
    # Usage (rs):  get_autoNet_yields.py rs  <version> <mode> <num_workers>
    # Usage (mp):  get_autoNet_yields.py mp  <version> <mode> <num_workers> <pruner> <pruning> <paths_version>
    #   pruner  : batch | single
    #   pruning : prune | noprune
    source      = sys.argv[1]        # rs | mp
    version     = sys.argv[2]        # autonet version
    mode        = sys.argv[3]        # sbd | iter
    num_workers = int(sys.argv[4])

    if source not in ("rs", "mp"):
        raise ValueError(f"Unknown SOURCE '{source}'. Use 'rs' or 'mp'.")

    if mode == "sbd":
        compute_yield = compute_yield_sbd
    elif mode == "iter":
        compute_yield = compute_yield_iter
    else:
        raise ValueError(f"Unknown MODE '{mode}'. Use 'sbd' or 'iter'.")

    if source == "rs":
        autonet_path = f"data/networks/autonets_rs_P_v{version}.pkl"
        output_path  = f"data/yields/yields_auto_rs_P_v{version}_{mode}.pkl"
    else:  # mp
        pruner        = sys.argv[5]       # batch | single
        pruning       = sys.argv[6]       # prune | noprune
        paths_version = sys.argv[7]       # minpath dataset version
        prune_suffix  = "P" if pruning == "prune" else "NP"
        autonet_path  = f"data/networks/autonets_mp_{pruner}_{prune_suffix}_pv{paths_version}_v{version}.pkl"
        output_path   = f"data/yields/yields_auto_mp_{pruner}_{prune_suffix}_pv{paths_version}_v{version}_{mode}.pkl"

    with open(autonet_path, "rb") as f:
        AutoNets = pickle.load(f)

    num_nets = len(AutoNets)
    print(f"Loaded {num_nets} networks from {autonet_path}")

    E_yields = np.zeros(num_nets)
    B_yields = np.zeros(num_nets)
    viability = np.zeros(num_nets, dtype=bool)

    start = time.time()

    print(f"Using {num_workers} parallel workers")
    with mp.Pool(processes=num_workers) as pool:
        for i, (E_yield, B_yield, status) in enumerate(
                pool.imap(compute_yield, AutoNets, chunksize=64)):
            E_yields[i] = E_yield
            B_yields[i] = B_yield
            viability[i] = status

            if (i + 1) % 500 == 0:
                processed_ratio = (i + 1) / num_nets
                viable_ratio = np.sum(viability[:i+1]) / (i + 1)
                print(f"  Processed {i + 1}/{num_nets} ({processed_ratio:.2%}), viable: {np.sum(viability[:i+1])}/{i + 1} ({viable_ratio:.2%})")

    elapsed = time.time() - start
    valid = np.sum(viability)
    print(f"\nCompleted in {elapsed:.2f} seconds")
    print(f"Valid networks (all precursors produced): {valid}/{num_nets}")

    import os
    os.makedirs("data/yields", exist_ok=True)
    with open(output_path, "wb") as f:
        pickle.dump((E_yields, B_yields, viability), f)
    print(f"Saved yields to {output_path}")