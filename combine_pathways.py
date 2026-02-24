import numpy as np
import time
from prune_check import isAllCoreProduced

def buildAutonomousNetwork(pathway_list, rxnMat, prodMat, sumRxnVec, 
                           nutrientSet, Currency, coreTBPs, rng=None):
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

    # Iterative pruning
    currSatRxnVec = np.copy(satRxnVec)
    removed = 0
    while True:
        currRxns = np.nonzero(currSatRxnVec)[0]
        removable = 0

        for rxn in rng.permutation(currRxns):
            if isAllCoreProduced(rxn, currSatRxnVec, rxnMat, prodMat, sumRxnVec, 
                                 nutrientSet, Currency, coreTBPs):
                currSatRxnVec[rxn] = 0
                removable += 1
                removed += 1

        if removable == 0:
            print("No more removable reactions → terminating.")
            print(f'Final network size = {len(currRxns)}, removed {removed} reactions')
            break
    return currRxns