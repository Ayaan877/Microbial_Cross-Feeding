import numpy as np
from prune_check import isCoreProduced

def randMinNetwork(satRxnVec, rxnMat, prodMat, sumRxnVec,
                   coreTBP, nutrientSet, Currency, rng=None, init_frac=0.5):
    """
    Wrapper for batch-based pruning followed by single-reaction cleanup.
    """
    if rng is None:
        rng = np.random.default_rng()

    currSatRxnVec = np.copy(satRxnVec)
    fail_count = 0
    max_fails = 3

    batch_frac = init_frac

    while True:
        currSatRxns = np.nonzero(currSatRxnVec)[0]
        n_curr = len(currSatRxns)

        if n_curr == 0:
            break

        batch_size = max(1, int(n_curr * batch_frac))
        # print(f"[batch] Current size={n_curr}, batch_size={batch_size}")

        # ------------------------------------------------
        # Batchwise Reaction Removal Phase
        # ------------------------------------------------
        if batch_size > 1:
            batch = rng.choice(currSatRxns, size=batch_size, replace=False)

            if isCoreProduced(batch, currSatRxnVec, rxnMat, prodMat, 
                              sumRxnVec, nutrientSet, Currency, coreTBP):

                currSatRxnVec[batch] = 0
                fail_count = 0

            else:
                fail_count += 1
                if fail_count >= max_fails:
                    batch_frac /= 1.1
                    fail_count = 0

        # ------------------------------------------------
        # Single Reaction Systematic Removal Phase
        # ------------------------------------------------
        else:
            removed_any = False

            for rxn in rng.permutation(currSatRxns):
                if isCoreProduced([rxn], currSatRxnVec, rxnMat, prodMat,
                                  sumRxnVec, nutrientSet, Currency, coreTBP):

                    currSatRxnVec[rxn] = 0
                    removed_any = True

            if not removed_any:
                print("No more removable reactions → terminating.", flush=True)
                print(f'Minimal network size = {len(currSatRxns)}', flush=True)
                break
    
    return np.nonzero(currSatRxnVec)[0]