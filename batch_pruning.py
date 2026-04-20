import numpy as np
from prune_check import isCoreProduced

def randMinNetwork(satRxnVec, rxnMat, prodMat, sumRxnVec,
                   coreTBP, nutrientSet, Currency, rng=None, init_frac=0.5):
    """
    Wrapper for batch-based pruning followed by single-reaction cleanup.
    """
    # print(f"Batch pruning network...", flush=True)
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
                # print(f'Minimal network size = {len(currSatRxns)}', flush=True)
                break
    
    return np.nonzero(currSatRxnVec)[0]



def alt_randMinNetwork(satRxnVec, rxnMat, prodMat, sumRxnVec,
                   coreTBP, nutrientSet, Currency, donated_met, 
                   rng=None, init_frac=0.5):
    """
    Wrapper for batch-based pruning followed by single-reaction cleanup,
    while preserving dependence on donated_met.

    A removal is accepted only if:
      1. All protected mets are still producible WITH donated_met (viability).
      2. Not all protected mets are producible WITHOUT donated_met (dependency).
    """
    print(f"Batch pruning network while preserving dependence on intermediate {donated_met}...", flush=True)
    if rng is None:
        rng = np.random.default_rng()

    augmented_nutrients = list(nutrientSet) + [donated_met]
    base_nutrients = nutrientSet

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

        # ------------------------------------------------
        # Batchwise Reaction Removal Phase
        # ------------------------------------------------
        if batch_size > 1:
            batch = rng.choice(currSatRxns, size=batch_size, replace=False)

            viable   = isCoreProduced(batch, currSatRxnVec, rxnMat, prodMat,
                                      sumRxnVec, augmented_nutrients, Currency, coreTBP)
            dependent = not isCoreProduced(batch, currSatRxnVec, rxnMat, prodMat,
                                           sumRxnVec, base_nutrients, Currency, coreTBP)

            if viable and dependent:
                currSatRxnVec[batch] = 0
                fail_count = 0
            else:
                fail_count += 1
                if fail_count >= max_fails:
                    batch_frac /= 1.05
                    fail_count = 0

        # ------------------------------------------------
        # Single Reaction Systematic Removal Phase
        # ------------------------------------------------
        else:
            removed_any = False

            for rxn in rng.permutation(currSatRxns):
                viable   = isCoreProduced([rxn], currSatRxnVec, rxnMat, prodMat,
                                          sumRxnVec, augmented_nutrients, Currency, coreTBP)
                dependent = not isCoreProduced([rxn], currSatRxnVec, rxnMat, prodMat,
                                               sumRxnVec, base_nutrients, Currency, coreTBP)

                if viable and dependent:
                    currSatRxnVec[rxn] = 0
                    removed_any = True

            if not removed_any:
                # print(f'Minimal network size = {len(currSatRxns)}', flush=True)
                break
    
    return np.nonzero(currSatRxnVec)[0]