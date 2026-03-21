import numpy as np

def markSatMetsRxns(rxnProc, rxnMat, prodMat, sumRxnVec, nutrientSet, Currency):
    """
    Takes in a bunch of reactions and using the KEGG provided chemistry, 
    markes all reactions and metabolites in the list of reactions that are 
    'satisfied', i.e. which can be reached via simply a seed set of the 
    given nutrients and currency metabolites.

    RETURNS:

        satMetVec, satRxnVec: the current sets of metabolites and reactions
                          that are said to be satisfied.
    """
    n_rxns, n_mets = rxnMat.shape

    seeds = np.array(nutrientSet + Currency)

    # Only operate on candidate reactions to avoid full-matrix multiplies.
    active_rxns = np.nonzero(rxnProc)[0]

    if len(active_rxns) == 0:
        satMetVec = np.zeros(n_mets)
        satMetVec[seeds] = 1
        return satMetVec, np.zeros(n_rxns)

    # Extract submatrices for active reactions only.
    sub_rxnMat = rxnMat[active_rxns]
    sub_prodMat = prodMat[active_rxns]
    sub_sumRxnVec = sumRxnVec[active_rxns]

    # Compress metabolite dimension: keep only columns involved in any
    # active reaction plus the seed metabolites.
    seed_mask = np.zeros(n_mets, dtype=bool)
    seed_mask[seeds] = True
    involved_mets = np.nonzero(np.any(sub_rxnMat, axis=0) | np.any(sub_prodMat, axis=0) | seed_mask)[0]

    comp_rxnMat = sub_rxnMat[:, involved_mets]
    comp_prodMat_T = sub_prodMat[:, involved_mets].T
    
    # Map seed indices into compressed metabolite space.
    met_to_local = np.empty(n_mets, dtype=np.intp)
    met_to_local[involved_mets] = np.arange(len(involved_mets))

    n_involved = len(involved_mets)
    comp_satMetVec = np.zeros(n_involved)
    comp_satMetVec[met_to_local[seeds]] = 1

    comp_satRxnVec = np.zeros(len(active_rxns))

    while True:
        old_sub = comp_satRxnVec.copy()

        # Marking first reactions, then metabolites, iteratively.
        comp_satRxnVec = (comp_rxnMat @ comp_satMetVec == sub_sumRxnVec) * 1
        comp_satMetVec = (comp_prodMat_T @ comp_satRxnVec + comp_satMetVec > 0) * 1

        # Checking if all satisfied nodes have been marked.
        if np.array_equal(old_sub, comp_satRxnVec):
            break

    # Map back to full-size vectors.
    satMetVec = np.zeros(n_mets)
    satMetVec[involved_mets] = comp_satMetVec

    satRxnVec = np.zeros(n_rxns)
    satRxnVec[active_rxns] = comp_satRxnVec

    return satMetVec, satRxnVec