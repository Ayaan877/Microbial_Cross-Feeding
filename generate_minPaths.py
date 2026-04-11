from datetime import datetime
from reverse_scope import giveRevScope
from multiprocessing import Pool
import numpy as np
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
                             nutrientSet, Currency, n_variants, n_cores, randMinNetwork):
    '''
    Generates minimal subgraphs for each core molecule, using parallel computation.
    Returns: A list of unique pathways from the medium to the precursor. 
    '''
    
    satMets, satRxns = giveRevScope(rxnMat, prodMat, sumRxnVec, nutrientSet, Currency, target)

    unique_nets = set()
    attempts = 0
    max_attempts = 50

    while len(unique_nets) < n_variants and attempts < max_attempts:
        attempts += 1
        seeds = np.random.randint(0, 10**9, size=n_cores)

        variant_args = [(satRxns, rxnMat, prodMat, sumRxnVec,
                        target, nutrientSet, Currency, seed, randMinNetwork)
                        for seed in seeds]
        
        start = time.time()
        with Pool(processes=n_cores) as pool:
            new_nets = pool.map(single_variant, variant_args)
        elapsed = time.time() - start

        for net in new_nets:
            unique_nets.add(tuple(sorted(net))) 

        print(f"[{datetime.now().strftime('%H:%M:%S')}] Attempt {attempts}: {elapsed:.4f}s - {len(unique_nets)}/{n_variants} unique networks")

    return [np.array(net) for net in list(unique_nets)[:n_variants]]