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

def _isLimiting(tRct, tRxn, m, S, reactants):
    for oRct in reactants:
        if (m[tRct] * S[tRxn, oRct] / S[tRxn, tRct]) > m[oRct]:
            return True


def _giveLimitingCurrency(r, tRxn):
    return np.where(r[tRxn] == max(r[tRxn][np.where(r[tRxn] < 0.0)]))[0][0]


# ---------------------------------------------------------------------------
# Core algorithm: split_by_demand.py logic with core-checking bug fixed
# ---------------------------------------------------------------------------

def splitByDemand_alt(stoich_matrix, rxnMat, prodMat, sumRxnVec, rho, pi,
                      nutrientSet, Energy, Currency, Core, orgRxns):
    """
    split_by_demand.py logic with the original core-checking bug restored:
      isCoreProduced[Core] = 1  (bugged: marks all core mets produced at once)

    Key behavioural differences vs calculate_autoNet_yield.py:
      - Picks the first NON-limiting reactant (sbd.py: not isLimiting)
      - No nutrient-share persistence across rounds
      - No sumRxnVec == 0 filtering on procRxnVec
      - Returns (-1.0, -1.0, False) for non-viable networks
    """
    runningE, runningB = 0.0, 0.0

    nMets      = len(np.transpose(stoich_matrix))
    activeRxns = np.array(orgRxns, dtype=int)
    nActive    = len(activeRxns)
    isCoreProduced = np.zeros(nMets)

    metState = np.zeros(nMets)
    metState[Currency + nutrientSet] = 1

    r    = np.copy(rho[activeRxns])
    S    = np.copy(stoich_matrix[activeRxns])
    rMat = np.copy(rxnMat[activeRxns])
    pMat = np.copy(prodMat[activeRxns])
    sumRxnVecActive = sumRxnVec[activeRxns]

    procRxnVec = ((np.dot(rMat, metState != 0) - sumRxnVecActive) == 0) * 1

    isChecked = np.zeros(nActive)

    mask = np.abs(np.sum(r, axis=0)) != 0
    shareMatrix = np.zeros(S.shape, dtype=float)
    shareMatrix[:, np.where(mask)[0]] = ((r * metState)[:, mask] /
                                         np.abs(np.sum(r, axis=0))[mask])
    shareMatrix[:, Currency] = -1

    while procRxnVec.any():
        prodState = np.zeros(nMets)

        for thisRxn in np.where(procRxnVec)[0]:
            allowedRct = []
            isChecked[thisRxn] = 1

            rs        = np.where(rMat[thisRxn])[0]
            ps        = np.where(pMat[thisRxn])[0]
            reactants = [tR for tR in rs if tR not in Currency]
            products  = [tP for tP in ps if tP not in Currency]

            # Pick the first non-limiting reactant (sbd.py logic).
            for thisReactant in reactants:
                if not _isLimiting(thisReactant, thisRxn,
                                   shareMatrix[thisRxn], S, reactants):
                    allowedRct.append(thisReactant)
                    limRct = deepcopy(thisReactant)
                    break

            if not allowedRct:
                limRct = _giveLimitingCurrency(r, thisRxn)

            for thisMet in products:
                ratio = S[thisRxn, thisMet] / S[thisRxn, limRct]
                prodState[thisMet] += shareMatrix[thisRxn, limRct] * ratio

            mets = np.append(rs, ps)
            for thisMet in mets[np.where(np.in1d(mets, np.array(Core + Energy)))]:
                ratio = S[thisRxn, thisMet] / S[thisRxn, limRct]

                if thisMet in Energy:
                    runningE += shareMatrix[thisRxn, limRct] * ratio
                elif thisMet in Core:
                    if thisMet in ps:
                        isCoreProduced[Core] = 1
                    runningB += shareMatrix[thisRxn, limRct] * ratio

            for thisMet in reactants:
                ratio = S[thisRxn, thisMet] / S[thisRxn, limRct]
                shareMatrix[thisRxn, thisMet] -= shareMatrix[thisRxn, limRct] * ratio

        # Redistribution: update ALL columns with remaining demand (sbd.py approach).
        r[np.where(isChecked)] = 0
        mask = np.abs(np.sum(r, axis=0)) != 0
        shareMatrix[:, np.where(mask)[0]] = ((r * prodState)[:, mask] /
                                             np.abs(np.sum(r, axis=0))[mask])
        shareMatrix[:, Currency] = -1
        # Note: no nutrient-share persistence across rounds (sbd.py logic)

        procRxnVec = ((np.dot(rMat, np.sum(shareMatrix, axis=0) != 0) -
                       sumRxnVecActive) == 0) * 1
        procRxnVec[np.where(isChecked)] = 0
        # Note: no sumRxnVec == 0 filtering (sbd.py logic)

    if isCoreProduced[Core].all():
        return runningE, runningB, True
    else:
        return -1.0, -1.0, False


def compute_yield_alt(net):
    return splitByDemand_alt(
        stoich_matrix, rxnMat, prodMat,
        sumRxnVec, rho, pi, nutrientSet,
        Energy, Currency, Core, net)


if __name__ == "__main__":

    # Args supplied by PBS script (mirrors run_autonomous_yields.pbs)
    # Usage (rs): get_alt_autoNet_yields.py rs  <version> <mode> <num_workers>
    # Usage (mp): get_alt_autoNet_yields.py mp  <version> <mode> <num_workers> <pruner> <pruning> <paths_version>
    #   mode is accepted for PBS compatibility but ignored (only one algorithm here)
    source      = sys.argv[1]        # rs | mp
    version     = sys.argv[2]        # autonet version
    mode       = sys.argv[3]        # accepted for compatibility, not used
    num_workers = int(sys.argv[4])

    if source not in ("rs", "mp"):
        raise ValueError(f"Unknown SOURCE '{source}'. Use 'rs' or 'mp'.")

    if source == "rs":
        autonet_path = f"data/networks/autonets_rs_P_v{version}.pkl"
        output_path  = f"data/yields/yields_auto_rs_P_v{version}_alt.pkl"
    else:
        pruner        = sys.argv[5]   # batch | single
        pruning       = sys.argv[6]   # prune | noprune
        paths_version = sys.argv[7]   # minpath dataset version
        prune_suffix  = "P" if pruning == "prune" else "NP"
        autonet_path  = f"data/networks/autonets_mp_{pruner}_{prune_suffix}_pv{paths_version}_v{version}.pkl"
        output_path   = f"data/yields/yields_auto_mp_{pruner}_{prune_suffix}_pv{paths_version}_v{version}_alt.pkl"

    with open(autonet_path, "rb") as f:
        AutoNets = pickle.load(f)

    num_nets = len(AutoNets)
    print(f"Loaded {num_nets} networks from {autonet_path}")

    E_yields  = np.zeros(num_nets)
    B_yields  = np.zeros(num_nets)
    viability = np.zeros(num_nets, dtype=bool)

    start = time.time()

    print(f"Using {num_workers} parallel workers")
    with mp.Pool(processes=num_workers) as pool:
        for i, (E_yield, B_yield, status) in enumerate(
                pool.imap(compute_yield_alt, AutoNets, chunksize=64)):
            E_yields[i]  = E_yield
            B_yields[i]  = B_yield
            viability[i] = status

            if (i + 1) % 500 == 0:
                processed_ratio = (i + 1) / num_nets
                viable_ratio    = np.sum(viability[:i+1]) / (i + 1)
                print(f"  Processed {i + 1}/{num_nets} ({processed_ratio:.2%}), "
                      f"viable: {np.sum(viability[:i+1])}/{i + 1} ({viable_ratio:.2%})")

    elapsed = time.time() - start
    valid   = np.sum(viability)
    print(f"\nCompleted in {elapsed:.2f} seconds")
    print(f"Valid networks (all precursors produced): {valid}/{num_nets}")

    import os
    os.makedirs("data/yields", exist_ok=True)
    with open(output_path, "wb") as f:
        pickle.dump((E_yields, B_yields, viability), f)
    print(f"Saved yields to {output_path}")
