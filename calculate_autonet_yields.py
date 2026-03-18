'''
AUTONET YIELD CALCULATOR
Loads autonomous networks and calculates energy/biomass yields for each.
Supports parallel computation via multiprocessing.
'''
from calculate_yield import *
from load_data import *
import time
import multiprocessing as mp


def _compute_yield(net):
    '''Worker function for parallel yield calculation.
    Uses module-level globals from load_data (shared via fork COW).'''
    return splitByDemand(
        stoich_matrix, rxnMat, prodMat,
        sumRxnVec, rho, pi, nutrientSet,
        Energy, Currency, Core, net)


def autonetYields(autonet_path, num_workers):
    '''
    Loads a set of autonomous networks and calculates
    the energy and biomass yields for each network.

    Args:
        autonet_path: Path to the pickled autonomous networks file.
        num_workers: Number of parallel workers (1 = serial).

    Returns:
        E_yields: Energy yield (ATP+NADPH) per nutrient molecule
        B_yields: Biomass yield (precursors) per nutrient molecule
        viability: Boolean array of viability
    '''
    with open(autonet_path, "rb") as f:
        AutoNets = pickle.load(f)

    num_nets = len(AutoNets)
    print(f"Loaded {num_nets} networks from {autonet_path}")

    E_yields = np.zeros(num_nets)
    B_yields = np.zeros(num_nets)
    viability = np.zeros(num_nets, dtype=bool)

    start = time.time()

    if num_workers > 1:
        print(f"Using {num_workers} parallel workers")
        with mp.Pool(processes=num_workers) as pool:
            for i, (E_yield, B_yield, status) in enumerate(
                    pool.imap(_compute_yield, AutoNets, chunksize=64)):
                E_yields[i] = E_yield
                B_yields[i] = B_yield
                viability[i] = status

                if (i + 1) % 500 == 0:
                    elapsed_so_far = time.time() - start
                    rate = (i + 1) / elapsed_so_far
                    eta = (num_nets - i - 1) / rate
                    print(f"  Processed {i + 1}/{num_nets} "
                          f"({rate:.1f} nets/s, ETA {eta/60:.1f} min)")
    else:
        print("Using serial computation")
        for i, net in enumerate(AutoNets):
            E_yield, B_yield, status = splitByDemand(
                stoich_matrix, rxnMat, prodMat,
                sumRxnVec, rho, pi, nutrientSet,
                Energy, Currency, Core, net)
            E_yields[i] = E_yield
            B_yields[i] = B_yield
            viability[i] = status

            if (i + 1) % 100 == 0:
                print(f"  Processed {i + 1}/{num_nets} networks...")

    elapsed = time.time() - start
    valid = np.sum(viability)
    print(f"\nCompleted in {elapsed:.2f} seconds")
    print(f"Valid networks (all precursors produced): {valid}/{num_nets}")

    return E_yields, B_yields, viability