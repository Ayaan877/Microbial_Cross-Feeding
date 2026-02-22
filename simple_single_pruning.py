import numpy as np
from prune_check import isCoreProduced

### Serial Algorithm ###

def randMinNetwork(satRxnVec, rxnMat, prodMat, sumRxnVec,
                   coreTBP, nutrientSet, Currency, rng=None):
    """
    Takes in a set of satisfied reactions (belonging to a subgraph) and prunes
    it down to a minimal subgraph by randomly removing singly removable reactions.
    """

    if rng is None:
        rng = np.random.default_rng()

    currSatRxnVec = np.copy(satRxnVec)
    count = 0

    while True:
        count += 1
        # Keeping track of what the graph currently looks like.
        currSatRxns = np.nonzero(currSatRxnVec)[0]
        print(f"[Sweep {count}] Current size: {len(currSatRxns)} reactions", flush=True)

        removed_any = False

        # Randomize order for stochastic minimal networks
        removalOrder = rng.permutation(currSatRxns)

        n = len(removalOrder)
        print('Checking reactions in random order for removability...', flush=True)
        for i, remRxn in enumerate(removalOrder, 1):
            if i == 1 or i % 100 == 0 or i == n:
                print(f"[Sweep {count}] {100*i/n:.1f}%", flush=True)

        for remRxn in removalOrder:
            # print(f"Checking if reaction {remRxn} is removable...", flush=True)

            if isCoreProduced(remRxn, currSatRxnVec,
                              rxnMat, prodMat,
                              sumRxnVec, nutrientSet,
                              Currency, coreTBP):

                currSatRxnVec[remRxn] = 0
                removed_any = True
        
        # If we completed a full pass with no removals → minimal
        if not removed_any:
            print("[randMinSubnet] Finished. Minimal subset achieved.", flush=True)
            return np.nonzero(currSatRxnVec)[0]