import numpy as np
from satisfiability_check import markSatMetsRxns

def get_intermediates(net, rxnMat, prodMat, sumRxnVec, nutrientSet, Currency, excluded):
    """
    All metabolites produced by network, excluding precursors/currency/nutrients/energy
    """
    satRxnVec = np.zeros(rxnMat.shape[0], dtype=int)
    satRxnVec[net] = 1
    satMets, satRxns = markSatMetsRxns(satRxnVec, rxnMat, prodMat,
                                 sumRxnVec, nutrientSet, Currency)
    produced = set(np.nonzero(satMets)[0])
    intermediates = np.array(sorted(produced - set(excluded)))
    return intermediates


def get_byproducts(net, rxnMat, prodMat, sumRxnVec, nutrientSet, Currency, excluded):
    """
    Metabolites produced by the network that are not consumed as reactants
    by any reaction in the network, excluding precursors/currency/nutrients/energy.
    """
    intermediates = get_intermediates(net, rxnMat, prodMat, sumRxnVec,
                                               nutrientSet, Currency, excluded)
    reactVec = np.logical_or.reduce(rxnMat[net]) if len(net) > 0 else np.zeros(rxnMat.shape[1], dtype=bool)
    byproducts = np.array([m for m in intermediates if not reactVec[m]])
    return byproducts


def get_candidates(net, rxnMat, prodMat, sumRxnVec, nutrientSet, Currency, 
                   Core, use_byproducts):
    
    excluded = sorted(set(Currency + Core + nutrientSet))

    if use_byproducts:
        candidates = get_byproducts(net, rxnMat, prodMat, sumRxnVec,
                                       nutrientSet, Currency, excluded)
    else:
        candidates = get_intermediates(net, rxnMat, prodMat, sumRxnVec,
                                          nutrientSet, Currency, excluded)
    return candidates

if __name__ == "__main__":
    import pickle
    from load_data import *

    # ── Config ──────────────────────────────────────────────────────────────
    AUTONET_SUBDIR  = "autonets_rs_av1"   # autonets_{source}_av{version}
    AUTONET_FILE    = "P"                  # P (rs is always pruned)
    NET_IDX_A       = 2395
    NET_IDX_B       = 1965
    # ──────────────────────────────────────────────────────────────────
    from directory_paths import resolve_autonet_path
    with open(resolve_autonet_path(AUTONET_SUBDIR, AUTONET_FILE), "rb") as f:
        all_autonets = pickle.load(f)

    net_A = all_autonets[NET_IDX_A]
    net_B = all_autonets[NET_IDX_B]

    candidates_A = get_candidates(net_A, rxnMat, prodMat, sumRxnVec,
                                   nutrientSet, Currency, Core, use_byproducts=False)
    candidates_B = get_candidates(net_B, rxnMat, prodMat, sumRxnVec,
                                   nutrientSet, Currency, Core, use_byproducts=False)
    common = np.array(sorted(list(set(candidates_A) & set(candidates_B))), dtype=int)
    