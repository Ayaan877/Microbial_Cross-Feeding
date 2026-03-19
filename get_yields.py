from calculate_yield import *
from load_data import *
import time
import multiprocessing as mp
import pickle
import sys

def compute_yield(net):
    return splitByDemand(
        stoich_matrix, rxnMat, prodMat,
        sumRxnVec, rho, pi, nutrientSet,
        Energy, Currency, Core, net)

if __name__ == "__main__":
    mode = sys.argv[1]
    minimal = sys.argv[2]
    dataset = sys.argv[3]

    autonet_dir = f"AutoNets{dataset}_{mode}_{minimal}"
    autonet_data = f"AutoNets{dataset}_{mode}.pkl"
    autonet_path = f"{autonet_dir}/{autonet_data}"
    num_workers = 32

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
                elapsed_so_far = time.time() - start
                rate = (i + 1) / elapsed_so_far
                eta = (num_nets - i - 1) / rate
                print(f"  Processed {i + 1}/{num_nets} "
                        f"({rate:.1f} nets/s, ETA {eta/60:.1f} min)")

    elapsed = time.time() - start
    valid = np.sum(viability)
    print(f"\nCompleted in {elapsed:.2f} seconds")
    print(f"Valid networks (all precursors produced): {valid}/{num_nets}")

    with open(f"{autonet_dir}/Yields_{autonet_data}", "wb") as f:
        pickle.dump((E_yields, B_yields, viability), f)