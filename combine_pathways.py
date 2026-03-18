import numpy as np
import time
from prune_check import isAllCoreProduced

def buildAutonomousNetwork(pathway_list, rxnMat, prodMat, sumRxnVec, 
                           nutrientSet, Currency, coreTBPs, prune, rng=None):
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
        currSatRxnVec = np.copy(satRxnVec)

        while True:
            currSatRxns = np.nonzero(currSatRxnVec)[0]
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