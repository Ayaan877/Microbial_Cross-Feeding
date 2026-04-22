import numpy as np
from satisfiability_check import markSatMetsRxns

def verify_autonomy(net, rxnMat, prodMat, sumRxnVec, nutrientSet, Currency, Core):
    """
    Check that any given network is viable i.e. can produce all cores with a 
    specified nutrient set (which may include donated metabolites).
    """
    rxnVec = np.zeros(rxnMat.shape[0], dtype=int)
    rxnVec[net] = 1
    satMets, _ = markSatMetsRxns(rxnVec, rxnMat, prodMat,
                                 sumRxnVec, nutrientSet, Currency)

    missing_cores = [c for c in Core if not satMets[c]]

    return len(missing_cores) == 0, missing_cores