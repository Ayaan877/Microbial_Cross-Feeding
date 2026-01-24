import numpy as np
from prune_check import isCoreProduced

def randMinSubnetwork(satRxnVec, rxnMat, prodMat, sumRxnVec,
                        coreProdRxns, coreTBP, nutrientSet, Currency,
                        init_frac=0.5, min_size=5):
    """
    Batch-based pruning: start by removing a large random batch (e.g., 50%),
    shrinking batch size on failure, down to a minimum batch fraction.
    Then do a final single-reaction pruning sweep.
    """

    currSatRxnVec = np.copy(satRxnVec)
    n_rxns = len(currSatRxnVec)
    fail_count = 0
    max_fails = 3

    # -----------------------
    # 1. BATCH REMOVAL PHASE
    # -----------------------
    batch_frac = init_frac

    while True:
        currRxns = np.nonzero(currSatRxnVec)[0]
        if len(currRxns) == 0:
            break

        batch_size = max(1, int(len(currRxns) * batch_frac))
        print(f"[batch] Current size={len(currRxns)}, batch_size={batch_size}")

        # Select random batch
        batch = np.random.choice(currRxns, size=batch_size, replace=False)

        # Try removing batch
        if isCoreProduced(batch, currSatRxnVec, rxnMat, prodMat, sumRxnVec,
                          coreProdRxns, nutrientSet, Currency, coreTBP):
            print(f"[batch] Removing batch of {batch_size}")
            currSatRxnVec[batch] = 0
            fail_count = 0
        else:
            fail_count += 1
            print(f"[batch] Could not remove batch of {batch_size}, trying again...")
            # Failed; reduce batch size
            if fail_count >= max_fails:
                batch_frac /= 1.1
                print(f"[batch] Batch failed → decreasing batch_frac to {batch_frac:.4f}")
                fail_count = 0

                if batch_size <= min_size:
                    print("[batch] Minimum batch size reached → going to cleanup.")
                    break

            # if fail_count >= max_fails:
            #     print("[batch] Minimum batch size reached → moving to final cleanup.")
            #     break

    # ------------------------------------
    # 2. FINAL SINGLE-REACTION CLEANUP
    # ------------------------------------
    print("[cleanup] Starting final single-reaction sweep...")

    while True:
        changed = False
        currRxns = np.nonzero(currSatRxnVec)[0]

        if len(currRxns) == 0:
            break

        for rxn in currRxns:
            if isCoreProduced([rxn], currSatRxnVec, rxnMat, prodMat, sumRxnVec,
                              coreProdRxns, nutrientSet, Currency, coreTBP):
                print(f"[cleanup] Removing leftover singly-removable reaction {rxn}")
                currSatRxnVec[rxn] = 0
                changed = True

        if not changed:
            print("[cleanup] No more removable single reactions → minimality reached.")
            break

    # Return final minimal set
    return np.nonzero(currSatRxnVec)[0]


# def randMinSubnetwork(satRxnVec, rxnMat, prodMat, sumRxnVec,
#                         coreProdRxns, coreTBP, nutrientSet, Currency,
#                         initial_fraction=0.5):
#     """
#     Batch-removal minimal pruning **without** checking singly-removable reactions.

#     Strategy:
#         - Start with batch size = 50% of current reactions.
#         - Randomly choose a batch of that size.
#         - If feasible, remove it.
#         - If not, halve the batch and retry.
#         - Stop when a batch of size 1 cannot be removed.
#     """

#     currSat = np.copy(satRxnVec)

#     while True:
#         currRxns = np.nonzero(currSat)[0]
#         n_curr = len(currRxns)

#         print(f"[batch] Current size: {n_curr} reactions")

#         # If only 1 reaction left, cannot prune further
#         if n_curr <= 1:
#             print("[batch] Only one reaction left → minimal set reached.")
#             return currRxns

#         # Initial batch size = 50% of all current reactions
#         batch_size = max(1, int(n_curr * initial_fraction))

#         # Shuffle the reaction list so sampling is random
#         np.random.shuffle(currRxns)

#         success = False
#         while batch_size > 0:

#             # Random batch from all current reactions
#             batch = np.random.choice(currRxns, size=batch_size, replace=False)

#             print(f"[batch] Trying batch of size {batch_size}...")

#             if isCoreProduced(batch, currSat, rxnMat, prodMat, sumRxnVec,
#                               coreProdRxns, nutrientSet, Currency, coreTBP):
#                 print(f"[batch] Batch of size {batch_size} succeeded → pruning.")
#                 currSat[batch] = 0
#                 success = True
#                 break

#             # Shrink the batch and try again
#             batch_size //= 2
#             print(f"[batch] Batch failed → reducing to {batch_size}")

#         # If we reach batch_size = 0 or removing 1 fails → minimal
#         if not success:
#             print("[batch] No further batch can be removed → minimal set reached.")
#             return np.nonzero(currSat)[0]
        
#----------------------------------------------------------------------------------------------------------

# def randMinSubnetwork(satRxnVec, rxnMat, prodMat, sumRxnVec, 
#                   coreProdRxns, coreTBP, nutrientSet, Currency):
#     """
#     Takes in a set of satisfied reactions (belonging to a subgraph) and prunes
#     it down to a minimal subgraph by randomly removing singly removable reactions.
#     """
#     currSatRxnVec = np.copy(satRxnVec)

#     while True:

#         currSatRxns = np.nonzero(currSatRxnVec)[0]
#         print(f"[randMinSubnet] Current size: {len(currSatRxns)} reactions")

#         # --- NEW: pick a batch of candidates to try removing ---
#         # Example strategy: randomly permute all current reactions
#         # and attempt removing the first k in a batch.
#         # You can adjust k depending on aggressiveness.
#         k = max(1, len(currSatRxns) // 10)   # remove 10% at once
#         batchCandidates = np.random.permutation(currSatRxns)[:k]

#         print(f"   Trying batch removal of size {len(batchCandidates)}")

#         # Try removing them all at once
#         testVec = currSatRxnVec.copy()
#         testVec[batchCandidates] = 0

#         # Check if core is still producible
#         canRemoveBatch = isCoreProduced(
#             batchCandidates,                # not used in this context
#             currSatRxnVec, 
#             rxnMat, prodMat,
#             sumRxnVec,
#             coreProdRxns,
#             nutrientSet,
#             Currency,
#             coreTBP
#         )

#         if canRemoveBatch:
#             # The batch is valid → accept the removals
#             print(f"   Batch accepted. Removed {len(batchCandidates)} reactions.")
#             currSatRxnVec = testVec
#             continue
#         else:
#             # Batch too big → try smaller batch
#             print("   Batch rejected. Trying smaller batch...")

#             # Try removing one-by-one from that batch
#             removed_any = False
#             for remRxn in batchCandidates:
#                 testVec = currSatRxnVec.copy()
#                 testVec[remRxn] = 0

#                 if isCoreProduced(
#                     remRxn, testVec, rxnMat, prodMat, sumRxnVec,
#                     coreProdRxns, nutrientSet, Currency, coreTBP
#                 ):
#                     print(f"      Removed reaction {remRxn}")
#                     currSatRxnVec[remRxn] = 0
#                     removed_any = True

#             # If not even one reaction could be removed → done
#             if not removed_any:
#                 print("[randMinSubnet] Finished. No more batch or single removals possible.")
#                 return np.nonzero(currSatRxnVec)[0]

