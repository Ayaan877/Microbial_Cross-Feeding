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

    tempSatMets, tempSatRxns = prunedSatsMets(remRxns, satRxns, rxnMat, prodMat, sumRxnVec, 
                                              nutrientSet, Currency)
    
    # Checking if this still produces the core molecule.
    return bool(tempSatMets[coreTBP])

#-------------------------------------------------------------------------

def isAllCoreProduced(remRxns, satRxns, rxnMat, prodMat, sumRxnVec,
                      nutrientSet, Currency, coreTBPs):

    tempSatMets, tempSatRxns = prunedSatsMets(remRxns, satRxns, rxnMat, prodMat, sumRxnVec, 
                                              nutrientSet, Currency)

    return all(tempSatMets[t] for t in coreTBPs)