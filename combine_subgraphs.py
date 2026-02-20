# import numpy as np
# import itertools
# from multiprocessing import Pool, cpu_count
# from prune_check import isAllCoreProduced
# from satisfiability_check import markSatMetsRxns

# def process_combination(args):

#     (combo, n_rxns, rxnMat, prodMat, sumRxnVec, 
#      nutrientSet, Currency, coreTBPs, seed) = args

#     rng = np.random.default_rng(seed)

#     # 1. Union reactions
#     combined_rxns = set()
#     for pathway in combo:
#         combined_rxns.update(pathway)

#     combined_rxns = np.array(list(combined_rxns), dtype=int)

#     # 2. Build reaction vector
#     satRxnVec = np.zeros(n_rxns, dtype=int)
#     satRxnVec[combined_rxns] = 1

#     # 3. Check feasibility
#     satMetVec, satRxnVec = markSatMetsRxns(satRxnVec, rxnMat, prodMat, sumRxnVec, 
#                                            nutrientSet, Currency)

#     if not np.all(satMetVec[coreTBPs] == 1):
#         return None

#     # 4. Prune while preserving ALL targets
#     currSatRxnVec = np.copy(satRxnVec)

#     while True:

#         currRxns = np.nonzero(currSatRxnVec)[0]
#         removable = []

#         for rxn in currRxns:
#             if isAllCoreProduced(rxn, currSatRxnVec, rxnMat, prodMat, 
#                                  sumRxnVec, nutrientSet, Currency, coreTBPs):
#                 removable.append(rxn)

#         if len(removable) == 0:
#             break

#         removal_order = rng.permutation(removable)

#         for rxn in removal_order:
#             if isAllCoreProduced(rxn, currSatRxnVec, rxnMat, prodMat, sumRxnVec, nutrientSet, Currency, coreTBPs):
#                 currSatRxnVec[rxn] = 0

#     final_rxns = tuple(sorted(np.nonzero(currSatRxnVec)[0]))
#     return final_rxns


# def buildAutonomousNetwork(paths, rxnMat, prodMat, sumRxnVec, 
#                            nutrientSet, Currency, coreTBPs, nprocs=4):

#     # if nprocs is None:
#     #     nprocs = cpu_count()

#     n_rxns = rxnMat.shape[0]

#     all_combos = list(itertools.product(*paths))

#     print(f"Total combinations: {len(all_combos)}")
#     print(f"Using {nprocs} processes")

#     worker_args = [(combo, n_rxns, rxnMat, prodMat, sumRxnVec, 
#                     nutrientSet, Currency, coreTBPs, i) 
#                     for i, combo in enumerate(all_combos)]

#     with Pool(processes=nprocs) as pool:
#         results = pool.map(process_combination, worker_args)

#     minimal_networks = []
#     seen = set()

#     for res in results:
#         if res is None:
#             continue
#         if res not in seen:
#             seen.add(res)
#             minimal_networks.append(np.array(res))

#     return minimal_networks

import numpy as np
import itertools
from prune_check import isAllCoreProduced
from satisfiability_check import markSatMetsRxns


def buildAutonomousNetwork(
        pathways_per_target,   # list of length 8, each element = list of 4 rxn lists
        rxnMat,
        prodMat,
        sumRxnVec,
        nutrientSet,
        Currency,
        coreTBPs,              # array of 8 target metabolite indices
        rng=None,
        verbose=True
    ):
    """
    For each combination of one pathway per target:
        - Take union of reactions
        - Prune redundant reactions
        - Ensure all targets remain producible

    Returns:
        list of minimal reaction index arrays
    """

    if rng is None:
        rng = np.random.default_rng()

    n_rxns = rxnMat.shape[0]

    minimal_networks = []
    seen_networks = set()
    # Progress counters
    total_combos = int(np.prod([len(p) for p in pathways_per_target]))
    print(f"Total combinations to process: {total_combos}")
    found_count = 0

    # Generate all combinations (4^8 = 65536)
    for combo_idx, combo in enumerate(itertools.product(*pathways_per_target)):

        if verbose and combo_idx % 1000 == 0:
            print(f"Processing combination {combo_idx}")

        # -------------------------------------------------
        # 1. Take union of reactions across 8 pathways
        # -------------------------------------------------
        combined_rxns = set()
        for pathway in combo:
            combined_rxns.update(pathway)

        combined_rxns = np.array(list(combined_rxns), dtype=int)

        # -------------------------------------------------
        # 2. Build reaction vector
        # -------------------------------------------------
        satRxnVec = np.zeros(n_rxns, dtype=int)
        satRxnVec[combined_rxns] = 1

        # -------------------------------------------------
        # 3. Check initial feasibility
        # -------------------------------------------------
        satMetVec, satRxnVec = markSatMetsRxns(
            satRxnVec, rxnMat, prodMat, sumRxnVec,
            nutrientSet, Currency
        )

        if not np.all(satMetVec[coreTBPs] == 1):
            if verbose:
                print(f"Combo {combo_idx}: invalid (core targets not all producible), skipping")
            continue  # skip invalid combos

        # -------------------------------------------------
        # 4. Prune redundancies while preserving ALL targets
        # -------------------------------------------------
        currSatRxnVec = np.copy(satRxnVec)
        if verbose:
            print(f"Combo {combo_idx}: starting prune with {int(currSatRxnVec.sum())} reactions")

        while True:
            currRxns = np.nonzero(currSatRxnVec)[0]

            removable = []

            for rxn in currRxns:
                print(f"Checking if reaction {rxn} is removable...", flush=True)
                if isAllCoreProduced(
                        rxn,
                        currSatRxnVec,
                        rxnMat,
                        prodMat,
                        sumRxnVec,
                        nutrientSet,
                        Currency,
                        coreTBPs
                    ):
                    removable.append(rxn)

            if len(removable) == 0:
                break

            removal_order = rng.permutation(removable)

            for rxn in removal_order:
                if isAllCoreProduced(
                        rxn,
                        currSatRxnVec,
                        rxnMat,
                        prodMat,
                        sumRxnVec,
                        nutrientSet,
                        Currency,
                        coreTBPs
                    ):
                    print(f"Removing reaction {rxn}...", flush=True)
                    currSatRxnVec[rxn] = 0

        final_rxns = tuple(sorted(np.nonzero(currSatRxnVec)[0]))

        # -------------------------------------------------
        # 5. Store unique minimal networks
        # -------------------------------------------------
        if final_rxns not in seen_networks:
            seen_networks.add(final_rxns)
            minimal_networks.append(np.array(final_rxns))
            found_count += 1
            if verbose and found_count % 10 == 0:
                print(f"Found {found_count} unique minimal networks so far (combo {combo_idx})")

    return minimal_networks
    