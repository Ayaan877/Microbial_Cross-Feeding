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

                print(f"Removing batch of {batch_size}")
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
    
    return currSatRxns


# def randMinNetwork(satRxnVec, rxnMat, prodMat, sumRxnVec,
#                    coreTBP, nutrientSet, Currency,
#                         init_frac=0.5, min_size=5):
#     """
#     Batch-based pruning: start by removing a large random batch (e.g., 50%),
#     shrinking batch size on failure, down to a minimum batch fraction.
#     Then do a final single-reaction pruning sweep.
#     """

#     currSatRxnVec = np.copy(satRxnVec)
#     fail_count = 0
#     max_fails = 3

#     # -----------------------
#     # 1. BATCH REMOVAL PHASE
#     # -----------------------
#     batch_frac = init_frac

#     while True:
#         currSatRxns = np.nonzero(currSatRxnVec)[0]
#         if len(currSatRxns) == 0:
#             break

#         batch_size = max(1, int(len(currSatRxns) * batch_frac))
#         print(f"[batch] Current size={len(currSatRxns)}, batch_size={batch_size}")

#         # Select random batch
#         batch = np.random.choice(currSatRxns, size=batch_size, replace=False)

#         # Try removing batch
#         if isCoreProduced(batch, currSatRxnVec, rxnMat, prodMat, sumRxnVec, 
#                           nutrientSet, Currency, coreTBP):
#             print(f"[batch] Removing batch of {batch_size}")
#             currSatRxnVec[batch] = 0
#             fail_count = 0
#         else:
#             fail_count += 1
#             # print(f"[batch] Could not remove batch of {batch_size}, trying again...")
#             # Failed; reduce batch size
#             if fail_count >= max_fails:
#                 batch_frac /= 1.1
#                 # print(f"[batch] Batch failed → decreasing batch_frac to {batch_frac:.4f}")
#                 fail_count = 0

#                 if batch_size <= min_size:
#                     print("[batch] Minimum batch size reached → going to cleanup.")
#                     break

#     # ------------------------------------
#     # 2. FINAL SINGLE-REACTION CLEANUP
#     # ------------------------------------
#     print("[cleanup] Starting final single-reaction sweep...")

#     while True:
#         changed = False
#         currSatRxns = np.nonzero(currSatRxnVec)[0]

#         if len(currSatRxns) == 0:
#             break

#         for rxn in currSatRxns:
#             if isCoreProduced([rxn], currSatRxnVec, rxnMat, prodMat, sumRxnVec,
#                               nutrientSet, Currency, coreTBP):
#                 # print(f"[cleanup] Removing leftover singly-removable reaction {rxn}")
#                 currSatRxnVec[rxn] = 0
#                 changed = True

#         if not changed:
#             print("[cleanup] No more removable single reactions → minimality reached.")
#             break

#     # Return final minimal set
#     return np.nonzero(currSatRxnVec)[0]
