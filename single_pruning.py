import numpy as np
from multiprocessing import Pool
from prune_check import isCoreProduced

### Parallel Algorithm ###

def check_removable(args):
    """
    Worker function for multiprocessing.
    Must be top-level (not nested) to be pickleable.
    """
    (remRxn, currSatRxnVec, rxnMat, prodMat,
     sumRxnVec, nutrientSet, Currency, coreTBP) = args

    return isCoreProduced(remRxn, currSatRxnVec, rxnMat, prodMat, 
                          sumRxnVec, nutrientSet, Currency, coreTBP)

def  randMinNetwork(satRxnVec, rxnMat, prodMat, sumRxnVec, 
                    coreTBP, nutrientSet, Currency, rng=None, nprocs=16):
    """
    Takes in a set of satisfied reactions (belonging to a subgraph) and prunes
    it down to a minimal subgraph by randomly removing singly removable reactions.
    """

    if rng is None:
        rng = np.random.default_rng()

    currSatRxnVec = np.copy(satRxnVec)

    with Pool(processes=nprocs) as pool:

        while True:

            currSatRxns = np.nonzero(currSatRxnVec)[0]
            print(f"[randMinSubnet] Current size: {len(currSatRxns)}", flush=True)

            # Parallel removal check for each reaction in the current subgraph
            args_iterable = [
                (remRxn, currSatRxnVec, rxnMat, prodMat,
                 sumRxnVec, nutrientSet, Currency, coreTBP)
                for remRxn in currSatRxns]

            canRemoveVec = pool.map(check_removable, args_iterable)
            canRemoveVec = np.array(canRemoveVec, dtype=int)

            removableRxns = currSatRxns[np.where(canRemoveVec)[0]]

            print(f"Removable this round: {len(removableRxns)}", flush=True)

            if len(removableRxns) == 0:
                print("[randMinSubnet] Finished. Minimal subset achieved.", flush=True)
                return currSatRxns

            # Sequential removal of singly removable reactions in random order
            removalOrder = rng.permutation(removableRxns)

            for remRxn in removalOrder:
                if isCoreProduced(remRxn, currSatRxnVec, rxnMat, prodMat,
                                  sumRxnVec, nutrientSet, Currency, coreTBP):
                    currSatRxnVec[remRxn] = 0

### Serial Algorithm ###

# def randMinNetwork(satRxnVec, rxnMat, prodMat, sumRxnVec, 
#                    coreTBP, nutrientSet, Currency, rng=None):
#     """
#     Takes in a set of satisfied reactions (belonging to a subgraph) and prunes
#     it down to a minimal subgraph by randomly removing singly removable reactions.
#     """
#     if rng is None:
#         rng = np.random.default_rng()
        
#     currSatRxnVec = np.copy(satRxnVec)

#     while True:
#         # Keeping track of what the graph currently looks like.
#         currSatRxns = np.nonzero(currSatRxnVec)[0]

#         print(f"[randMinSubnet] Current size: {len(currSatRxns)} reactions", flush=True)

#         # Marking out which reactions are singly removable.
#         # print(f"Checking reaction {remRxn} for removal...", flush=True)
#         canRemoveVec = np.array([isCoreProduced(remRxn, currSatRxnVec, rxnMat, prodMat, 
#                                                 sumRxnVec, nutrientSet, Currency, coreTBP) 
#                                                 for remRxn in currSatRxns]) * 1

#         # Keeping a list of what is removable.
#         removableMets = currSatRxns[ np.where( canRemoveVec ) ]

#         print(f"Removable this round: {len(removableMets)}", flush=True)

#         # If nothing can be removed, minimality condition satisfied. 
#         # Quitting with what we have
#         if len(removableMets) == 0:
#             print("[randMinSubnet] Finished. Minimal subset achieved.", flush=True)
#             return currSatRxns

#         # Randomly permuting the singly removable reactions.
#         removalOrder = rng.permutation(removableMets)

#         # Calling a vector of reaction that can be removed.
#         for remRxn in removalOrder:
#             if isCoreProduced(remRxn, currSatRxnVec, rxnMat, prodMat, 
#                               sumRxnVec, nutrientSet, Currency, coreTBP):
#                 currSatRxnVec[remRxn] = 0
#             # else:
#             #     break
