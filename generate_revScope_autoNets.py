import numpy as np
import pickle
import time
from datetime import datetime
from multiprocessing import Pool
from reverse_scope import giveRevScope
from batch_pruning import randMinNetwork


def prune_worker(args):
    """Multiprocessing worker: prune the shared reverse-scope network."""
    (satRxns, rxnMat, prodMat, sumRxnVec,
     coreTBPs, nutrientSet, Currency, seed) = args

    rng = np.random.default_rng(seed)
    return randMinNetwork(satRxns, rxnMat, prodMat, sumRxnVec,
                          coreTBPs, nutrientSet, Currency, rng=rng)

def generate_revScopeAutoNets(rxnMat, prodMat, sumRxnVec, nutrientSet,
                              Currency, coreTBPs, n_target=50000,
                              n_workers=32, batch_size=None,
                              save_path=None):
    """
    Generate `n_target` unique minimal autonomous networks by repeatedly
    batch-pruning the same reverse-scope subgraph with different seeds.
    """
    if batch_size is None:
        batch_size = n_workers

    satMets, satRxns = giveRevScope(rxnMat, prodMat, sumRxnVec,
                                    nutrientSet, Currency, coreTBPs)
    print(f"Reverse scope found {int(np.sum(satRxns))} reactions.")

    unique_nets = set()
    attempts = 0

    with Pool(processes=n_workers) as pool:
        while len(unique_nets) < n_target:
            attempts += 1
            seeds = np.random.randint(0, 2**31, size=batch_size)

            worker_args = [(satRxns, rxnMat, prodMat, sumRxnVec,
                            coreTBPs, nutrientSet, Currency, int(s))
                           for s in seeds]

            start = time.time()
            results = pool.map(prune_worker, worker_args)
            elapsed = time.time() - start

            for net in results:
                unique_nets.add(tuple(sorted(net)))

            print(f"[{datetime.now().strftime('%H:%M:%S')}] "
                  f"[Attempt {attempts}] {len(unique_nets)} unique autonets, {elapsed:.1f}s ")

    networks = [np.array(net) for net in unique_nets]
    print(f"Finished: {len(networks)} unique autonets in {attempts} attempts.")

    if save_path is not None:
        with open(save_path, "wb") as f:
            pickle.dump(networks, f)
        print(f"Saved to {save_path}")

    return networks


if __name__ == "__main__":
    import sys
    from load_data import *

    n_workers = int(sys.argv[1]) if len(sys.argv) > 1 else 32

    networks = generate_revScopeAutoNets(
        rxnMat, prodMat, sumRxnVec, nutrientSet, Currency, Core,
        n_target=50000, n_workers=n_workers,
        save_path="revScope_autonets.pkl")

    print(f"Final: {len(networks)} unique autonets")

