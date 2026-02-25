import numpy as np
import itertools
from multiprocessing import cpu_count, Pool
from combine_pathways import buildAutonomousNetwork

def process_network(args):
    (combo, rxnMat, prodMat, sumRxnVec,
     nutrientSet, Currency, coreTBPs, seed) = args

    rng = np.random.default_rng(seed)

    network = buildAutonomousNetwork(combo, rxnMat, prodMat, sumRxnVec, nutrientSet, 
                                     Currency, coreTBPs, rng=rng)

    return tuple(sorted(network))

def allAutonomousNetworks(all_paths, rxnMat, prodMat, sumRxnVec,
                                   nutrientSet, Currency, coreTBPs,
                                   n_processes=None, chunk_size=100):

    if n_processes is None:
        n_processes = cpu_count()

    print(f"Using {n_processes} processes")

    combo_generator = itertools.product(*all_paths)

    pool = Pool(processes=n_processes)

    seen_networks = set()
    minimal_networks = []

    job_iter = ((combo, rxnMat, prodMat, sumRxnVec,
         nutrientSet, Currency, coreTBPs, i)
        for i, combo in enumerate(combo_generator))

    for result in pool.imap_unordered(process_network, job_iter, chunksize=chunk_size):
        if result not in seen_networks:
            seen_networks.add(result)
            minimal_networks.append(np.array(result))

    pool.close()
    pool.join()

    print(f"Total unique autonomous networks: {len(minimal_networks)}")

    return minimal_networks