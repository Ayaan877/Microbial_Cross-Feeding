from datetime import datetime
from reverse_scope import giveRevScope
from multiprocessing import Pool
import numpy as np
import pickle
import time


def single_variant(args):
    '''
    Multiprocessing worker function for each variant
    '''
    (satRxns, rxnMat, prodMat, sumRxnVec,
     target, nutrientSet, Currency, seed, randMinNetwork) = args

    rng = np.random.default_rng(seed)

    return randMinNetwork(satRxns, rxnMat, prodMat, sumRxnVec,
                          target, nutrientSet, Currency, rng=rng)


def generate_pruned_networks(target, rxnMat, prodMat, sumRxnVec,
                             nutrientSet, Currency, n_workers, randMinNetwork,
                             save_path=None, max_attempts=500,
                             plateau_window=5, plateau_threshold=2):

    satMets, satRxns = giveRevScope(rxnMat, prodMat, sumRxnVec,
                                    nutrientSet, Currency, target)

    unique_nets = set()

    attempts_list = []
    unique_counts = []

    attempt = 0

    with Pool(processes=n_workers) as pool:
        while attempt < max_attempts:

            attempt += 1
            seeds = np.random.randint(0, 10**9, size=n_workers)

            variant_args = [(satRxns, rxnMat, prodMat, sumRxnVec,
                             target, nutrientSet, Currency,
                             seed, randMinNetwork)
                            for seed in seeds]

            start = time.time()

            new_nets = pool.map(single_variant, variant_args)

            elapsed = time.time() - start

            for net in new_nets:
                unique_nets.add(tuple(sorted(net)))

            current_count = len(unique_nets)

            attempts_list.append(attempt)
            unique_counts.append(current_count)

            print(f"[{datetime.now().strftime('%H:%M:%S')}] "
                  f"Attempt {attempt}: {elapsed:.4f}s - "
                  f"{current_count} unique networks")

            if save_path is not None:
                results = {"networks": [np.array(net) for net in unique_nets],
                           "attempts": attempts_list,
                           "unique_counts": unique_counts}
                with open(save_path, "wb") as f:
                    pickle.dump(results, f)

            if len(unique_counts) > plateau_window:
                recent_growth = unique_counts[-1] - unique_counts[-plateau_window]
                if recent_growth <= plateau_threshold:
                    print("Unique network discovery has plateaued. Stopping...")
                    break

    return {"networks": [np.array(net) for net in unique_nets], 
            "attempts": attempts_list, 
            "unique_counts": unique_counts}