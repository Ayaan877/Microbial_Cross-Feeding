import numpy as np
import pickle
import itertools
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
                                   n_processes=None, chunk_size=100,
                                   save_path=None, save_interval=100):

    if n_processes is None:
        n_processes = 32

    print(f"Using {n_processes} processes")

    combo_generator = itertools.product(*all_paths)

    pool = Pool(processes=n_processes)

    seen_networks = set()
    minimal_networks = []
    processed = 0

    job_iter = ((combo, rxnMat, prodMat, sumRxnVec,
         nutrientSet, Currency, coreTBPs, prune, i)
        for i, combo in enumerate(combo_generator))

    for result in pool.imap_unordered(process_network, job_iter, chunksize=chunk_size):
        if result not in seen_networks:
            seen_networks.add(result)
            minimal_networks.append(np.array(result))

        processed += 1
        if save_path is not None and processed % save_interval == 0:
            with open(save_path, "wb") as f:
                pickle.dump(minimal_networks, f)
            print(f"Checkpoint: {processed} processed, {len(minimal_networks)} unique networks saved")

    pool.close()
    pool.join()

    if save_path is not None:
        with open(save_path, "wb") as f:
            pickle.dump(minimal_networks, f)

    print(f"Total unique autonomous networks: {len(minimal_networks)}")

    return minimal_networks