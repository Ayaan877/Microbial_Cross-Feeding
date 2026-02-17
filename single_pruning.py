import numpy as np
from prune_check import isCoreProduced

def randMinNetwork(satRxnVec, rxnMat, prodMat, sumRxnVec, 
                   coreTBP, nutrientSet, Currency, rng=None):
    """
    Takes in a set of satisfied reactions (belonging to a subgraph) and prunes
    it down to a minimal subgraph by randomly removing singly removable reactions.
    """
    if rng is None:
        rng = np.random.default_rng()
        
    currSatRxnVec = np.copy(satRxnVec)

    while True:
        # Keeping track of what the graph currently looks like.
        currSatRxns = np.nonzero(currSatRxnVec)[0]

        print(f"[randMinSubnet] Current size: {len(currSatRxns)} reactions")

        # Marking out which reactions are singly removable.
        for remRxn in rng.permutation(currSatRxns):
            print(f"Checking reaction {remRxn} for removal...")
            canRemoveVec = np.array([isCoreProduced(remRxn, currSatRxnVec, rxnMat, prodMat, 
                                                    sumRxnVec, nutrientSet, Currency, coreTBP)]) * 1

        # Keeping a list of what is removable.
        removableMets = currSatRxns[ np.where( canRemoveVec ) ]

        print(f"Removable this round: {len(removableMets)}")

        # If nothing can be removed, minimality condition satisfied. 
        # Quitting with what we have now.
        if not removableMets.any():
            print("[randMinSubnet] Finished. Minimal subset achieved.")
            return np.nonzero(currSatRxnVec)[0]

        # Randomly permuting the singly removable reactions.
        removalOrder = rng.permutation(removableMets)

        # Calling a vector of reaction that can be removed.
        for remRxn in removalOrder:
            if isCoreProduced(remRxn, currSatRxnVec, rxnMat, prodMat, 
                              sumRxnVec, nutrientSet, Currency, coreTBP):
                currSatRxnVec[remRxn] = 0
            else:
                break
