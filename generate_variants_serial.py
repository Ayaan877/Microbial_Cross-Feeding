from datetime import datetime
from reverse_scope import giveRevScope
import numpy as np
import time

def generate_pruned_networks(target, rxnMat, prodMat, sumRxnVec,
                             nutrientSet, Currency, n_variants, n_cores, randMinNetwork):
    '''
    Generates minimal subgraphs for each core molecule, using serial computation.
    Returns: A list of unique pathways from the medium to the precursor. 
    '''

    print("Running reverse scope...")
    satMets, satRxns = giveRevScope(rxnMat, prodMat, sumRxnVec, nutrientSet, Currency, target)

    unique_nets = set()
    attempts = 0
    max_attempts = 50

    while len(unique_nets) < n_variants and attempts < max_attempts:

        attempts += 1

        seed = np.random.randint(0, 10**9)
        rng = np.random.default_rng(seed)

        start = time.time()

        net = randMinNetwork(satRxns, rxnMat, prodMat, sumRxnVec,
                             target, nutrientSet, Currency, rng=rng)

        elapsed = time.time() - start

        unique_nets.add(tuple(sorted(net)))

        print(f"[{datetime.now().strftime('%H:%M:%S')}] Attempt {attempts}: {elapsed:.4f}s - {len(unique_nets)}/{n_variants} unique networks", flush=True)

    return [np.array(net) for net in list(unique_nets)[:n_variants]]