from reverse_scope import giveRevScope
from batch_pruning import randMinNetwork

def revScope_autonet(rxnMat, prodMat, sumRxnVec,
                            nutrientSet, Currency, coreTBPs):
    """
    Return one minimal autonomous network (numpy array of reaction indices).
    """

    satMets, satRxns = giveRevScope(rxnMat, prodMat, sumRxnVec,
                                    nutrientSet, Currency, coreTBPs)
    return randMinNetwork(satRxns, rxnMat, prodMat, sumRxnVec,
                          coreTBPs, nutrientSet, Currency)


if __name__ == "__main__":
    from load_data import *
    import numpy as np
    import time
    from satisfiability_check import markSatMetsRxns

    start = time.time()
    net = revScope_autonet(rxnMat, prodMat, sumRxnVec,
                                  nutrientSet, Currency, Core)
    satRxnVec = np.zeros(rxnMat.shape[0], dtype=int)
    satRxnVec[net] = 1
    satMets, satRxns = markSatMetsRxns(satRxnVec, rxnMat, prodMat, sumRxnVec,
                             nutrientSet, Currency)
    
    viable = all(satMets[c] for c in Core)
    
    print(f"Generated autonet with {len(net)} reactions: {net}")
    print(f"Network viability: {viable}")
    print(f"Time taken: {(time.time() - start)} seconds")
