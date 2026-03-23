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
                             min_unique=50000, max_attempts=500000,
                             n_processes=None, chunk_size=100,
                             save_path=None, save_interval=100,
                             seed=None):

    if n_processes is None:
        n_processes = 32

    print(f"Using {n_processes} processes")
    print(f"Target: {min_unique} unique networks, max {max_attempts} attempts")

    master_rng = np.random.default_rng(seed)
    n_targets = len(all_paths)
    path_counts = [len(paths) for paths in all_paths]
    print(f"Pathway counts per target: {path_counts}")

    def combo_generator():
        attempt = 0
        while attempt < max_attempts:
            combo = tuple(
                all_paths[t][master_rng.integers(path_counts[t])]
                for t in range(n_targets)
            )
            yield (combo, rxnMat, prodMat, sumRxnVec,
                   nutrientSet, Currency, coreTBPs, prune,
                   master_rng.integers(2**31))
            attempt += 1

    pool = Pool(processes=n_processes)

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

        if len(minimal_networks) >= min_unique:
            print(f"Reached {min_unique} unique networks after {processed} attempts")
            break

    pool.close()
    pool.join()

    if save_path is not None:
        with open(save_path, "wb") as f:
            pickle.dump(minimal_networks, f)

    print(f"Total: {processed} attempts, {len(minimal_networks)} unique autonomous networks")

    return minimal_networks
