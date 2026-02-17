from datetime import datetime
from reverse_scope import giveRevScope
from batch_pruning import randMinNetwork
from multiprocessing import Pool
import numpy as np
import time

#### PARALLEL COMPUTATION WITH UNIQUENESS ####

def single_variant(args):
    (satRxns, rxnMat, prodMat, sumRxnVec,
     target, Energy, Currency, seed) = args

    rng = np.random.default_rng(seed)

    return randMinNetwork(satRxns, rxnMat, prodMat, sumRxnVec,
                          target, Energy, Currency, rng=rng)

def generate_pruned_networks(target, rxnMat, prodMat, sumRxnVec,
                             Energy, Currency, n_variants, n_cores):

    satMets, satRxns = giveRevScope(
        rxnMat, prodMat, sumRxnVec,
        Energy, Currency, target
    )

    unique_nets = set()
    attempts = 0
    max_attempts = 50

    while len(unique_nets) < n_variants and attempts < max_attempts:
        attempts += 1
        seeds = np.random.randint(0, 10**9, size=n_cores)

        variant_args = [(satRxns, rxnMat, prodMat, sumRxnVec,
                        target, Energy, Currency, seed)
                        for seed in seeds]
        
        start = time.time()
        with Pool(processes=n_cores) as pool:
            new_nets = pool.map(single_variant, variant_args)
        elapsed = time.time() - start

        for net in new_nets:
            unique_nets.add(tuple(sorted(net))) 

        print(f"[{datetime.now().strftime('%H:%M:%S')}] Attempt {attempts}: {elapsed:.4f}s - {len(unique_nets)}/{n_variants} unique networks")

    return [np.array(net) for net in list(unique_nets)[:n_variants]]