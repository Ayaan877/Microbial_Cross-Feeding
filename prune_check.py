import numpy as np
from satisfiability_check import markSatMetsRxns

def prunedSatsMets(remRxns, satRxns, rxnMat, prodMat, sumRxnVec, 
                   nutrientSet, Currency):
    # Creating a temporary set of reactions with
    # some reactions pruned.
    tempSatRxns = np.copy(satRxns)
    tempSatRxns[remRxns] = 0
    
    # Calculating the marked set of reactions from the temporary set.
    return markSatMetsRxns(tempSatRxns, rxnMat, prodMat, sumRxnVec, 
                           nutrientSet, Currency)

#-------------------------------------------------------------------------

def isCoreProduced(remRxns, satRxns, rxnMat, prodMat, sumRxnVec, 
                    nutrientSet, Currency, coreTBP):
    """
    Check whether core metabolite(s) are still produced after removing
    reactions. coreTBP can be a single index or a list/array of indices.
    """
    tempSatMets, tempSatRxns = prunedSatsMets(remRxns, satRxns, rxnMat, prodMat, sumRxnVec, 
                                              nutrientSet, Currency)

    cores = np.atleast_1d(np.asarray(coreTBP)).ravel()
    return all(tempSatMets[c] for c in cores)