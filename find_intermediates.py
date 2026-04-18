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
                   Core, Energy, use_byproducts=False):
    
    excluded = sorted(set(Currency + Energy + Core + nutrientSet))

    if use_byproducts:
        candidates = get_byproducts(net, rxnMat, prodMat, sumRxnVec,
                                       nutrientSet, Currency, excluded)
    else:
        candidates = get_intermediates(net, rxnMat, prodMat, sumRxnVec,
                                          nutrientSet, Currency, excluded)
    return candidates

