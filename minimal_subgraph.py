import numpy as np
from prune_check import isCoreProduced

def randMinSubnet(satRxnVec, rxnMat, prodMat, sumRxnVec, 
                  coreProdRxns, coreTBP, nutrientSet, Currency):
    """
    Takes in a set of satisfied reactions (belonging to a subgraph) and prunes
    it down to a minimal subgraph by randomly removing singly removable reactions.
    """
    currSatRxnVec = np.copy(satRxnVec)

    while True:
        # Keeping track of what the graph currently looks like.
        currSatRxns = np.nonzero(currSatRxnVec)[0]

        print(f"[randMinSubnet] Current size: {len(currSatRxns)} reactions")

        # Marking out which reactions are singly removable.
        for remRxn in currSatRxns:
            print(f"   Checking reaction {remRxn} for removal...")
            canRemoveVec = np.array([isCoreProduced(remRxn, currSatRxnVec, rxnMat, prodMat, sumRxnVec, 
                                                    coreProdRxns, nutrientSet, Currency, coreTBP)]) * 1

        # Keeping a list of what is removable.
        removableMets = currSatRxns[ np.where( canRemoveVec ) ]

        print(f"   Removable this round: {len(removableMets)}")

        # If nothing can be removed, minimality condition satisfied. 
        # Quitting with what we have now.
        if not removableMets.any():
            print("[randMinSubnet] Finished. Minimal subset achieved.")
            return np.nonzero(currSatRxnVec)[0]

        # Randomly permuting the singly removable reactions.
        removalOrder = np.random.permutation(removableMets)

        # Calling a vector of reaction that can be removed.
        for remRxn in removalOrder:
            if isCoreProduced(remRxn, currSatRxnVec, rxnMat, prodMat, 
                              sumRxnVec, coreProdRxns, nutrientSet, 
                              Currency, coreTBP):
                currSatRxnVec[remRxn] = 0
            else:
                break
