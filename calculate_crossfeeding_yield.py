import numpy as np
from copy import deepcopy

def isLimiting(tRct, tRxn, m, S, reactants):
    for oRct in reactants:
        if (m[tRct] * S[tRxn, oRct] / S[tRxn, tRct]) > m[oRct]:
            return True

def giveLimitingCurrency(r, tRxn):
    return np.where(r[tRxn] == max(r[tRxn][np.where(r[tRxn] < 0.0)]))[0][0]

def splitByDemand_crossfeeding(stoich_matrix, rxnMat, prodMat, sumRxnVec,
                                  rho, pi, nutrientSet, Energy, Currency,
                                  Core, crossPair):
    '''
    Takes one cross-feeding pair and calculates individual energy and biomass
    yields by solving both networks together in two coupled compartments.

    Compartments are linked only by the designated exchanged metabolites:
        A_donated: A --> B
        B_donated: B --> A
    '''
    reqKeys = {'cross_A', 'cross_B', 'A_donated', 'B_donated'}
    missing = reqKeys.difference(crossPair.keys())
    if missing:
        missingStr = ', '.join(sorted(missing))
        raise KeyError(f'crossPair is missing required keys: {missingStr}')

    nRxns, nMets = np.shape(stoich_matrix)

    cross_A = np.array(crossPair['cross_A'], dtype = int)
    cross_B = np.array(crossPair['cross_B'], dtype = int)
    metA_to_B = int(crossPair['A_donated'])
    metB_to_A = int(crossPair['B_donated'])

    if metA_to_B < 0 or metA_to_B >= nMets:
        raise ValueError('A_donated index is outside stoichiometric bounds.')
    if metB_to_A < 0 or metB_to_A >= nMets:
        raise ValueError('B_donated index is outside stoichiometric bounds.')

    activeA = np.where(cross_A)[0] if cross_A.dtype == bool else cross_A
    activeB = np.where(cross_B)[0] if cross_B.dtype == bool else cross_B
    nA = len(activeA)
    nB = len(activeB)

    # Initializing yield counters and production flags for each network.
    runningEA, runningBA = 0.0, 0.0
    runningEB, runningBB = 0.0, 0.0
    isCoreProducedA = np.zeros(nMets)
    isCoreProducedB = np.zeros(nMets)

    fluxAtoB, fluxBtoA = 0.0, 0.0

    # Constructing metabolite state for coupled A/B compartments.
    metState = np.zeros(2 * nMets)
    seedMets = np.array(list(Currency) + list(nutrientSet), dtype = int)
    if len(seedMets) > 0:
        metState[np.unique(seedMets)] = 1
        metState[np.unique(seedMets + nMets)] = 1

    # Building coupled matrices: A subnet + B subnet + 2 exchange links.
    nPairRxns = nA + nB + 2
    S = np.zeros((nPairRxns, 2 * nMets))
    r = np.zeros((nPairRxns, 2 * nMets))
    rMatPair = np.zeros((nPairRxns, 2 * nMets))
    pMatPair = np.zeros((nPairRxns, 2 * nMets))
    sumRxnVecPair = np.zeros(nPairRxns)

    if nA > 0:
        S[:nA, :nMets] = stoich_matrix[activeA]
        r[:nA, :nMets] = rho[activeA]
        rMatPair[:nA, :nMets] = rxnMat[activeA]
        pMatPair[:nA, :nMets] = prodMat[activeA]
        sumRxnVecPair[:nA] = sumRxnVec[activeA]

    if nB > 0:
        S[nA:nA + nB, nMets:] = stoich_matrix[activeB]
        r[nA:nA + nB, nMets:] = rho[activeB]
        rMatPair[nA:nA + nB, nMets:] = rxnMat[activeB]
        pMatPair[nA:nA + nB, nMets:] = prodMat[activeB]
        sumRxnVecPair[nA:nA + nB] = sumRxnVec[activeB]

    # Adding exchange reaction rows.
    rxnAtoB = nA + nB
    rxnBtoA = nA + nB + 1

    S[rxnAtoB, metA_to_B] = -1
    S[rxnAtoB, nMets + metA_to_B] = 1
    r[rxnAtoB, metA_to_B] = -1
    rMatPair[rxnAtoB, metA_to_B] = 1
    pMatPair[rxnAtoB, nMets + metA_to_B] = 1
    sumRxnVecPair[rxnAtoB] = 1

    S[rxnBtoA, nMets + metB_to_A] = -1
    S[rxnBtoA, metB_to_A] = 1
    r[rxnBtoA, nMets + metB_to_A] = -1
    rMatPair[rxnBtoA, nMets + metB_to_A] = 1
    pMatPair[rxnBtoA, metB_to_A] = 1
    sumRxnVecPair[rxnBtoA] = 1

    # Figuring out which reactions can be performed at this step.
    procRxnVec = ((np.dot(rMatPair, metState != 0) - sumRxnVecPair) == 0) * 1
    procRxnVec[sumRxnVecPair == 0] = 0

    # Continuing calculation till no more reactions can be performed.
    isChecked = np.zeros(nPairRxns)

    # Computing which reaction gets what share of which reactants.
    mask = np.abs(np.sum(r, axis = 0)) != 0
    shareMatrix = np.zeros(S.shape, dtype=float)
    shareMatrix[:, np.where(mask)[0]] = ((r * metState)[:, mask] /
                                         np.abs(np.sum(r, axis = 0))[mask])

    currencyAB = list(Currency) + [curr + nMets for curr in Currency]
    if len(currencyAB) > 0:
        shareMatrix[:, currencyAB] = -1

    # Saving initial total demand for nutrients to maintain their share across rounds.
    totalInitialDemandPair = np.abs(np.sum(r, axis = 0))
    nutrientColsAB = list(nutrientSet) + [m + nMets for m in nutrientSet]
    trackedEnergy = set(Energy)
    trackedCore = set(Core)
    trackedMetsA = np.array(list(trackedCore.union(trackedEnergy)), dtype = int)
    trackedMetsB = trackedMetsA + nMets
    trackedMets = np.append(trackedMetsA, trackedMetsB)

    while procRxnVec.any():

        # Initializing the product metabolite state vector.
        prodState = np.zeros(2 * nMets)

        # Updating states after all accomplishable reactions.
        for thisRxn in np.where(procRxnVec)[0]:
            # Checking if found a usable reactant.
            allowedRct = []
            isChecked[thisRxn] = 1

            # Getting reactants and products of this reaction, except currency.
            rs, ps = np.where(rMatPair[thisRxn])[0], np.where(pMatPair[thisRxn])[0]
            reactants = [tR for tR in rs if tR not in currencyAB]
            products = [tP for tP in ps if tP not in currencyAB]

            # Checking for limiting reactants.
            for thisReactant in reactants:
                if isLimiting(thisReactant, thisRxn, shareMatrix[thisRxn], S, reactants):
                    allowedRct.append(thisReactant)
                    limRct = deepcopy(thisReactant)
                    break

            # If nothing is limiting, everything gets used.
            if not allowedRct:
                if reactants:
                    limRct = reactants[0]
                else:
                    limRct = giveLimitingCurrency(r, thisRxn)

            # Updating metabolite amounts post reaction.
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

                # Updating E and B for network A or B.
                if thisMet in trackedEnergy:
                    runningEA += shareMatrix[thisRxn, limRct] * ratio
                elif thisMet in [e + nMets for e in trackedEnergy]:
                    runningEB += shareMatrix[thisRxn, limRct] * ratio
                elif thisMet in trackedCore:
                    if thisMet in ps:
                        isCoreProducedA[thisMet] = 1
                    runningBA += shareMatrix[thisRxn, limRct] * ratio
                elif thisMet in [c + nMets for c in trackedCore]:
                    if thisMet in ps:
                        isCoreProducedB[thisMet - nMets] = 1
                    runningBB += shareMatrix[thisRxn, limRct] * ratio

            # Updating metabolite amounts post reaction.
            for thisMet in reactants:
                ratio = S[thisRxn, thisMet] / S[thisRxn, limRct]
                shareMatrix[thisRxn, thisMet] -= shareMatrix[thisRxn, limRct] * ratio

        # Redistributing produced metabolites among reactions by demand.
        r[np.where(isChecked)] = 0
        totalDemandPair = np.abs(np.sum(r, axis = 0))
        newProdCols = np.where((prodState != 0) & (totalDemandPair != 0))[0]
        if len(newProdCols) > 0:
            shareMatrix[:, newProdCols] = ((r * prodState)[:, newProdCols] /
                                           totalDemandPair[newProdCols])
        if len(currencyAB) > 0:
            shareMatrix[:, currencyAB] = -1

        # Maintaining nutrient shares across rounds using initial total demand.
        for col in nutrientColsAB:
            if totalInitialDemandPair[col] > 0:
                shareMatrix[:, col] = r[:, col] / totalInitialDemandPair[col]

        # Recalculating performable reactions.
        procRxnVec = ((np.dot(rMatPair, np.sum(shareMatrix, axis = 0) != 0) -
                       sumRxnVecPair) == 0) * 1
        procRxnVec[np.where(isChecked)] = 0
        procRxnVec[sumRxnVecPair == 0] = 0

    statusA = bool(isCoreProducedA[Core].all())
    statusB = bool(isCoreProducedB[Core].all())

    return {
        'E_A': float(runningEA),
        'B_A': float(runningBA),
        'viable_A': statusA,
        'E_B': float(runningEB),
        'B_B': float(runningBB),
        'viable_B': statusB,
        'pair_viable': bool(statusA and statusB),
        'flux_A_to_B': float(fluxAtoB),
        'flux_B_to_A': float(fluxBtoA),
    }
