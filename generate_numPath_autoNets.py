import numpy as np
import pickle
from multiprocessing import Pool
from combine_pathways import buildAutonomousNetwork

def process_network(args):
    (combo, rxnMat, prodMat, sumRxnVec,
     nutrientSet, Currency, coreTBPs, prune, seed) = args

    rng = np.random.default_rng(seed)

    network = buildAutonomousNetwork(combo, rxnMat, prodMat, sumRxnVec, nutrientSet,
                                     Currency, coreTBPs, prune=prune, rng=rng)

    return tuple(sorted(network))

def allAutonomousNetworks(all_paths, rxnMat, prodMat, sumRxnVec,
                             nutrientSet, Currency, coreTBPs, prune,
                             n_target, n_workers, chunk_size=100,
                             save_path=None, save_interval=100,
                             seed=None):

    print(f"Using {n_workers} processes")
    print(f"Target: {n_target} unique networks")

    master_rng = np.random.default_rng(seed)
    n_targets = len(all_paths)
    path_counts = [len(paths) for paths in all_paths]
    print(f"Pathway counts per target: {path_counts}")

    def combo_generator():
        while True:
            combo = tuple(
                all_paths[t][master_rng.integers(path_counts[t])]
                for t in range(n_targets)
            )
            yield (combo, rxnMat, prodMat, sumRxnVec,
                   nutrientSet, Currency, coreTBPs, prune,
                   master_rng.integers(2**31))

    pool = Pool(processes=n_workers)

    seen_networks = set()
    minimal_networks = []
    processed = 0

    for result in pool.imap_unordered(process_network, combo_generator(), chunksize=chunk_size):
        if result not in seen_networks:
            seen_networks.add(result)
            minimal_networks.append(np.array(result))

        processed += 1

        if processed % save_interval == 0:
            print(f"  {processed} processed, {len(minimal_networks)} unique")
            if save_path is not None:
                with open(save_path, "wb") as f:
                    pickle.dump(minimal_networks, f)

        if len(minimal_networks) >= n_target:
            print(f"Reached {n_target} unique networks after {processed} attempts")
            break

    pool.terminate()
    pool.join()

    if save_path is not None:
        with open(save_path, "wb") as f:
            pickle.dump(minimal_networks, f)

    print(f"Total: {processed} attempts, {len(minimal_networks)} unique autonomous networks")

    return minimal_networks
