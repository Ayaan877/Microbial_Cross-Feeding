import numpy as np
import time
from prune_check import isAllCoreProduced

def buildAutonomousNetwork(pathway_list, rxnMat, prodMat, sumRxnVec, 
                           nutrientSet, Currency, coreTBPs, prune, rng=None, init_frac=0.5):
    """
    pathway_list : list of 8 pathways (each pathway = list of reaction indices)
    coreTBPs     : array of 8 target metabolite indices

    Returns:
        minimal reaction index array that produces all 8 targets
        from nutrientSet + Currency and is minimal under reachability.
    """

    if rng is None:
        rng = np.random.default_rng()

    # Union of reactions from 8 pathways
    combined_rxns = set()
    for pathway in pathway_list:
        combined_rxns.update(pathway)

    combined_rxns = np.array(list(combined_rxns), dtype=int)
    print(f"Combined network size before pruning: {len(combined_rxns)}")

    satRxnVec = np.zeros(rxnMat.shape[0], dtype=int)
    satRxnVec[combined_rxns] = 1
    
    if prune:
        # Iterative pruning (batch + single cleanup)
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

            # Batch removal phase
            if batch_size > 1:
                batch = rng.choice(currSatRxns, size=batch_size, replace=False)

                if isAllCoreProduced(batch, currSatRxnVec, rxnMat, prodMat,
                                    sumRxnVec, nutrientSet, Currency, coreTBPs):
                    print(f"Removing batch of {batch_size}")
                    currSatRxnVec[batch] = 0
                    fail_count = 0
                else:
                    fail_count += 1
                    if fail_count >= max_fails:
                        batch_frac /= 1.1
                        fail_count = 0

            # Single reaction systematic removal phase
            else:
                removed_any = False

                for rxn in rng.permutation(currSatRxns):
                    if isAllCoreProduced(rxn, currSatRxnVec, rxnMat, prodMat,
                                        sumRxnVec, nutrientSet, Currency, coreTBPs):
                        currSatRxnVec[rxn] = 0
                        removed_any = True

                if not removed_any:
                    print("No more removable reactions → terminating.")
                    print(f'Final network size = {len(currSatRxns)}')
                    break

        return np.nonzero(currSatRxnVec)[0]
    else:
        return combined_rxns