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

def generate_minimal_autonets(rxnMat, prodMat, sumRxnVec, nutrientSet,
                              Currency, coreTBPs, n_target=50000,
                              n_workers=32, batch_size=None,
                              plateau_window=10, plateau_threshold=5,
                              save_path=None):
    """
    Generate many unique minimal autonomous networks by repeatedly
    batch-pruning the same reverse-scope subgraph with different seeds.

    Parameters
    ----------
    n_target          : stop after collecting this many unique networks
    n_workers         : number of parallel worker processes
    batch_size        : networks per Pool.map round (default: n_workers)
    plateau_window    : number of rounds to look back for plateau check
    plateau_threshold : stop if fewer than this many new unique networks
                        were found in the last plateau_window rounds
    save_path         : pickle checkpoint path (None = no checkpointing)
    """
    if batch_size is None:
        batch_size = n_workers

    # ------------------------------------------------------------------
    # Reverse scope (once)
    # ------------------------------------------------------------------
    satMets, satRxns = giveRevScope(rxnMat, prodMat, sumRxnVec,
                                    nutrientSet, Currency, coreTBPs)
    print(f"Reverse scope found {int(np.sum(satRxns))} reactions.")

    # ------------------------------------------------------------------
    # Parallel pruning
    # ------------------------------------------------------------------
    unique_nets = set()
    attempts_list = []
    unique_counts = []
    round_num = 0

    with Pool(processes=n_workers) as pool:
        while len(unique_nets) < n_target:
            round_num += 1
            seeds = np.random.randint(0, 2**31, size=batch_size)

            worker_args = [(satRxns, rxnMat, prodMat, sumRxnVec,
                            coreTBPs, nutrientSet, Currency, int(s))
                           for s in seeds]

            start = time.time()
            results = pool.map(prune_worker, worker_args)
            elapsed = time.time() - start

            for net in results:
                unique_nets.add(tuple(sorted(net)))

            current_count = len(unique_nets)
            attempts_list.append(round_num)
            unique_counts.append(current_count)

            print(f"[{datetime.now().strftime('%H:%M:%S')}] "
                  f"Round {round_num} ({batch_size} seeds, {elapsed:.1f}s): "
                  f"{current_count} unique autonets")

            if save_path is not None and round_num % 5 == 0:
                save_checkpoint(unique_nets, attempts_list,
                                 unique_counts, save_path)

            # Plateau detection
            if len(unique_counts) > plateau_window:
                recent_growth = (unique_counts[-1]
                                 - unique_counts[-plateau_window])
                if recent_growth <= plateau_threshold:
                    print("Discovery plateaued. Stopping.")
                    break

    if save_path is not None:
        save_checkpoint(unique_nets, attempts_list, unique_counts, save_path)

    print(f"Finished: {len(unique_nets)} unique autonets in {round_num} rounds.")
    return {"networks": [np.array(net) for net in unique_nets],
            "attempts": attempts_list,
            "unique_counts": unique_counts}


def save_checkpoint(unique_nets, attempts_list, unique_counts, save_path):
    results = {"networks": [np.array(net) for net in unique_nets],
               "attempts": attempts_list,
               "unique_counts": unique_counts}
    with open(save_path, "wb") as f:
        pickle.dump(results, f)


if __name__ == "__main__":
    import sys
    from load_data import *

    n_workers = int(sys.argv[1]) if len(sys.argv) > 1 else 32

    results = generate_minimal_autonets(
        rxnMat, prodMat, sumRxnVec, nutrientSet, Currency, Core,
        n_target=50_000, n_workers=n_workers,
        save_path="revScope_autonets.pkl")

    print(f"Final: {len(results['networks'])} unique autonets")

