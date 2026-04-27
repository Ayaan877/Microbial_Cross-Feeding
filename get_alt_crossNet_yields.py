import numpy as np
from copy import deepcopy
from load_data import *
import sys
import time
import multiprocessing as mp
import pickle


# ---------------------------------------------------------------------------
# Helper functions (identical to split_by_demand.py)
# ---------------------------------------------------------------------------

def isLimiting(tRct, tRxn, m, S, reactants):
    for oRct in reactants:
        if (m[tRct] * S[tRxn, oRct] / S[tRxn, tRct]) > m[oRct]:
            return True


def giveLimitingCurrency(r, tRxn):
    return np.where(r[tRxn] == max(r[tRxn][np.where(r[tRxn] < 0.0)]))[0][0]


# ---------------------------------------------------------------------------
# Core algorithm: split_by_demand.py logic applied to cross-feeding pairs,
# with the original core-checking bug restored.
# ---------------------------------------------------------------------------

def splitByDemand_crossfeeding_alt(stoich_matrix, rxnMat, prodMat, sumRxnVec,
                                   rho, pi, nutrientSet, Energy, Currency,
                                   Core, crossPair):
    """
    Coupled two-compartment extension of split_by_demand.py with the
    original core-checking bug restored.

    Key behavioural differences vs calculate_crossNet_yield.py:
      - Picks the first NON-limiting reactant (sbd.py: not isLimiting)
      - No nutrient-share persistence across rounds
      - No sumRxnVecPair == 0 filtering on procRxnVec
      - Redistribution updates ALL columns with remaining demand (sbd.py style)
    """
    reqKeys = {'cross_A', 'cross_B', 'A_donated', 'B_donated'}
    missing = reqKeys.difference(crossPair.keys())
    if missing:
        raise KeyError(f'crossPair is missing required keys: {", ".join(sorted(missing))}')

    nRxns, nMets = np.shape(stoich_matrix)

    cross_A   = np.array(crossPair['cross_A'], dtype=int)
    cross_B   = np.array(crossPair['cross_B'], dtype=int)
    metA_to_B = int(crossPair['A_donated'])
    metB_to_A = int(crossPair['B_donated'])

    if metA_to_B < 0 or metA_to_B >= nMets:
        raise ValueError('A_donated index is outside stoichiometric bounds.')
    if metB_to_A < 0 or metB_to_A >= nMets:
        raise ValueError('B_donated index is outside stoichiometric bounds.')

    activeA = np.where(cross_A)[0] if cross_A.dtype == bool else cross_A
    activeB = np.where(cross_B)[0] if cross_B.dtype == bool else cross_B
    nA, nB  = len(activeA), len(activeB)

    runningEA, runningBA = 0.0, 0.0
    runningEB, runningBB = 0.0, 0.0
    isCoreProducedA = np.zeros(nMets)
    isCoreProducedB = np.zeros(nMets)
    fluxAtoB, fluxBtoA = 0.0, 0.0

    # Seed both compartments with currency and nutrients.
    metState = np.zeros(2 * nMets)
    seedMets = np.array(list(Currency) + list(nutrientSet), dtype=int)
    if len(seedMets) > 0:
        metState[np.unique(seedMets)]         = 1
        metState[np.unique(seedMets + nMets)] = 1

    # Build coupled matrices: A subnet + B subnet + 2 exchange reactions.
    nPairRxns     = nA + nB + 2
    S             = np.zeros((nPairRxns, 2 * nMets))
    r             = np.zeros((nPairRxns, 2 * nMets))
    rMatPair      = np.zeros((nPairRxns, 2 * nMets))
    pMatPair      = np.zeros((nPairRxns, 2 * nMets))
    sumRxnVecPair = np.zeros(nPairRxns)

    if nA > 0:
        S[:nA, :nMets]        = stoich_matrix[activeA]
        r[:nA, :nMets]        = rho[activeA]
        rMatPair[:nA, :nMets] = rxnMat[activeA]
        pMatPair[:nA, :nMets] = prodMat[activeA]
        sumRxnVecPair[:nA]    = sumRxnVec[activeA]

    if nB > 0:
        S[nA:nA+nB, nMets:]        = stoich_matrix[activeB]
        r[nA:nA+nB, nMets:]        = rho[activeB]
        rMatPair[nA:nA+nB, nMets:] = rxnMat[activeB]
        pMatPair[nA:nA+nB, nMets:] = prodMat[activeB]
        sumRxnVecPair[nA:nA+nB]    = sumRxnVec[activeB]

    rxnAtoB = nA + nB
    rxnBtoA = nA + nB + 1

    S[rxnAtoB, metA_to_B]              = -1
    S[rxnAtoB, nMets + metA_to_B]      =  1
    r[rxnAtoB, metA_to_B]              = -1
    rMatPair[rxnAtoB, metA_to_B]       =  1
    pMatPair[rxnAtoB, nMets + metA_to_B] = 1
    sumRxnVecPair[rxnAtoB]             =  1

    S[rxnBtoA, nMets + metB_to_A]      = -1
    S[rxnBtoA, metB_to_A]              =  1
    r[rxnBtoA, nMets + metB_to_A]      = -1
    rMatPair[rxnBtoA, nMets + metB_to_A] = 1
    pMatPair[rxnBtoA, metB_to_A]       =  1
    sumRxnVecPair[rxnBtoA]             =  1

    procRxnVec = ((np.dot(rMatPair, metState != 0) - sumRxnVecPair) == 0) * 1
    # Note: no sumRxnVecPair == 0 filtering (sbd.py logic)

    isChecked = np.zeros(nPairRxns)

    mask = np.abs(np.sum(r, axis=0)) != 0
    shareMatrix = np.zeros(S.shape, dtype=float)
    shareMatrix[:, np.where(mask)[0]] = ((r * metState)[:, mask] /
                                         np.abs(np.sum(r, axis=0))[mask])

    currencyAB = list(Currency) + [c + nMets for c in Currency]
    if currencyAB:
        shareMatrix[:, currencyAB] = -1

    trackedEnergy = set(Energy)
    trackedCore   = set(Core)
    trackedMetsA  = np.array(list(trackedCore | trackedEnergy), dtype=int)
    trackedMetsB  = trackedMetsA + nMets
    trackedMets   = np.append(trackedMetsA, trackedMetsB)

    while procRxnVec.any():
        prodState = np.zeros(2 * nMets)

        for thisRxn in np.where(procRxnVec)[0]:
            allowedRct = []
            isChecked[thisRxn] = 1

            rs        = np.where(rMatPair[thisRxn])[0]
            ps        = np.where(pMatPair[thisRxn])[0]
            reactants = [tR for tR in rs if tR not in currencyAB]
            products  = [tP for tP in ps if tP not in currencyAB]

            # Pick the first non-limiting reactant (sbd.py logic).
            for thisReactant in reactants:
                if not isLimiting(thisReactant, thisRxn,
                                   shareMatrix[thisRxn], S, reactants):
                    allowedRct.append(thisReactant)
                    limRct = deepcopy(thisReactant)
                    break

            if not allowedRct:
                limRct = giveLimitingCurrency(r, thisRxn)

            for thisMet in products:
                ratio = S[thisRxn, thisMet] / S[thisRxn, limRct]
                prodState[thisMet] += shareMatrix[thisRxn, limRct] * ratio

            if thisRxn == rxnAtoB:
                fluxAtoB += -shareMatrix[thisRxn, limRct]
            elif thisRxn == rxnBtoA:
                fluxBtoA += -shareMatrix[thisRxn, limRct]

            mets = np.append(rs, ps)
            for thisMet in mets[np.where(np.isin(mets, trackedMets))]:
                ratio = S[thisRxn, thisMet] / S[thisRxn, limRct]

                if thisMet in trackedEnergy:
                    runningEA += shareMatrix[thisRxn, limRct] * ratio
                elif thisMet in [e + nMets for e in trackedEnergy]:
                    runningEB += shareMatrix[thisRxn, limRct] * ratio
                elif thisMet in trackedCore:
                    if thisMet in ps:
                        isCoreProducedA[Core] = 1
                    runningBA += shareMatrix[thisRxn, limRct] * ratio
                elif thisMet in [c + nMets for c in trackedCore]:
                    if thisMet in ps:
                        isCoreProducedB[Core] = 1
                    runningBB += shareMatrix[thisRxn, limRct] * ratio

            for thisMet in reactants:
                ratio = S[thisRxn, thisMet] / S[thisRxn, limRct]
                shareMatrix[thisRxn, thisMet] -= shareMatrix[thisRxn, limRct] * ratio

        # Redistribution: update ALL columns with remaining demand (sbd.py approach).
        r[np.where(isChecked)] = 0
        mask = np.abs(np.sum(r, axis=0)) != 0
        shareMatrix[:, np.where(mask)[0]] = ((r * prodState)[:, mask] /
                                             np.abs(np.sum(r, axis=0))[mask])
        if currencyAB:
            shareMatrix[:, currencyAB] = -1
        # Note: no nutrient-share persistence across rounds (sbd.py logic)

        procRxnVec = ((np.dot(rMatPair, np.sum(shareMatrix, axis=0) != 0) -
                       sumRxnVecPair) == 0) * 1
        procRxnVec[np.where(isChecked)] = 0
        # Note: no sumRxnVecPair == 0 filtering (sbd.py logic)

    statusA = bool(isCoreProducedA[Core].all())
    statusB = bool(isCoreProducedB[Core].all())

    return {
        'E_A':         float(runningEA),
        'B_A':         float(runningBA),
        'viable_A':    statusA,
        'E_B':         float(runningEB),
        'B_B':         float(runningBB),
        'viable_B':    statusB,
        'pair_viable': bool(statusA and statusB),
        'flux_A_to_B': float(fluxAtoB),
        'flux_B_to_A': float(fluxBtoA),
    }


def compute_crossfeeding_yield_alt(crossPair):
    result = splitByDemand_crossfeeding_alt(
        stoich_matrix, rxnMat, prodMat,
        sumRxnVec, rho, pi, nutrientSet,
        Energy, Currency, Core, crossPair)
    return (
        result['E_A'], result['B_A'], result['viable_A'],
        result['E_B'], result['B_B'], result['viable_B'],
        result['pair_viable'],
        result['flux_A_to_B'], result['flux_B_to_A'],
    )


if __name__ == "__main__":

    # Args supplied by PBS script
    # Usage (rs): get_alt_crossNet_yields.py rs  <autonet_id> <crossnet_id> <crossnet_type> <num_workers>
    # Usage (mp): get_alt_crossNet_yields.py mp  <autonet_id> <crossnet_id> <crossnet_type> <num_workers> <pruner> <pruning> <paths_version>
    source        = sys.argv[1]        # rs | mp
    autonet_id    = sys.argv[2]        # autonet version
    crossnet_id   = sys.argv[3]        # crossnet run version
    crossnet_type = sys.argv[4]        # byp | int
    num_workers   = int(sys.argv[5])

    if source not in ("rs", "mp"):
        raise ValueError(f"Unknown source '{source}'. Use 'rs' or 'mp'.")
    if crossnet_type not in ("byp", "int"):
        raise ValueError(f"Unknown crossnet_type '{crossnet_type}'. Use 'byp' or 'int'.")

    if source == "rs":
        crossnet_path = f"data/networks/crossnets_rs_P_v{autonet_id}_{crossnet_type}_v{crossnet_id}.pkl"
        output_path   = f"data/yields/yields_cross_rs_P_v{autonet_id}_{crossnet_type}_v{crossnet_id}_alt.pkl"
    else:
        pruner        = sys.argv[6]   # batch | single
        pruning       = sys.argv[7]   # prune | noprune
        paths_version = sys.argv[8]   # paths dataset version
        suffix        = "P" if pruning == "prune" else "NP"
        crossnet_path = f"data/networks/crossnets_mp_{pruner}_{suffix}_pv{paths_version}_v{autonet_id}_{crossnet_type}_v{crossnet_id}.pkl"
        output_path   = f"data/yields/yields_cross_mp_{pruner}_{suffix}_pv{paths_version}_v{autonet_id}_{crossnet_type}_v{crossnet_id}_alt.pkl"

    with open(crossnet_path, "rb") as f:
        CrossNets = pickle.load(f)

    num_pairs = len(CrossNets)
    print(f"Loaded {num_pairs} cross-feeding pairs from {crossnet_path}")

    E_A_yields  = np.zeros(num_pairs)
    B_A_yields  = np.zeros(num_pairs)
    viable_A    = np.zeros(num_pairs, dtype=bool)
    E_B_yields  = np.zeros(num_pairs)
    B_B_yields  = np.zeros(num_pairs)
    viable_B    = np.zeros(num_pairs, dtype=bool)
    pair_viable = np.zeros(num_pairs, dtype=bool)
    flux_A_to_B = np.zeros(num_pairs)
    flux_B_to_A = np.zeros(num_pairs)

    start = time.time()

    print(f"Using {num_workers} parallel workers")
    with mp.Pool(processes=num_workers) as pool:
        for i, (EA, BA, vA, EB, BB, vB, vPair, fAB, fBA) in enumerate(
                pool.imap(compute_crossfeeding_yield_alt, CrossNets, chunksize=64)):
            E_A_yields[i]  = EA
            B_A_yields[i]  = BA
            viable_A[i]    = vA
            E_B_yields[i]  = EB
            B_B_yields[i]  = BB
            viable_B[i]    = vB
            pair_viable[i] = vPair
            flux_A_to_B[i] = fAB
            flux_B_to_A[i] = fBA

            if (i + 1) % 500 == 0:
                processed_ratio = (i + 1) / num_pairs
                viable_ratio    = np.sum(pair_viable[:i+1]) / (i + 1)
                print(f"  Processed {i + 1}/{num_pairs} ({processed_ratio:.2%}), "
                      f"pair viable: {np.sum(pair_viable[:i+1])}/{i + 1} ({viable_ratio:.2%})")

    elapsed = time.time() - start
    valid   = np.sum(pair_viable)
    print(f"\nCompleted in {elapsed:.2f} seconds")
    print(f"Viable pairs (both organisms produce all precursors): {valid}/{num_pairs}")

    import os
    os.makedirs("data/yields", exist_ok=True)
    with open(output_path, "wb") as f:
        pickle.dump({
            'E_A':         E_A_yields,
            'B_A':         B_A_yields,
            'viable_A':    viable_A,
            'E_B':         E_B_yields,
            'B_B':         B_B_yields,
            'viable_B':    viable_B,
            'pair_viable': pair_viable,
            'flux_A_to_B': flux_A_to_B,
            'flux_B_to_A': flux_B_to_A,
        }, f)
    print(f"Saved yields to {output_path}")
