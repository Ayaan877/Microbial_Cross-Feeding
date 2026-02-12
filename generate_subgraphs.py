from reverse_scope import giveRevScope
from alt_minimal_subgraph import randMinNetwork
from multiprocessing import Pool

#### PARALLEL COMPUTATION ####

def single_variant(args):
    (satRxns, rxnMat, prodMat, sumRxnVec, target, Energy, Currency) = args

    return randMinNetwork(satRxns, rxnMat, prodMat, sumRxnVec, 
                            [], target, Energy, Currency)


def generate_pruned_networks(target, rxnMat, prodMat, sumRxnVec, Energy, 
                             Currency, n_variants, n_cores):
    """
    Generates minimal network variants in parallel.
    """

    # Compute non-minimal subgraph 
    satMets, satRxns = giveRevScope(rxnMat, prodMat, sumRxnVec, Energy, Currency, target)

    variant_args = [(satRxns, rxnMat, prodMat, sumRxnVec, target, 
                    Energy, Currency) for _ in range(n_variants)]

    with Pool(processes=n_cores) as pool:
        variants = pool.map(single_variant, variant_args)

    return variants

#### SERIES COMPUTATION ####

# def generate_pruned_networks(target, rxnMat, prodMat, sumRxnVec, Energy, Currency, n_variants=5):
    
#     satMets, satRxns = giveRevScope(rxnMat, prodMat, sumRxnVec, Energy, Currency, target)

#     with open("inv_met_map.pkl", "rb") as f:
#         inv_met_map = pickle.load(f)

#     target_id = inv_met_map[target]
#     variants = []
#     for _ in range(n_variants):
#         min_rxns = randMinNetwork(satRxns, rxnMat, prodMat, sumRxnVec, [], target, Energy, Currency)
#         variants.append(min_rxns)

#     with open(f"{target_id}_MinNetworks.pkl", "wb") as f:
#         pickle.dump(variants, f)

#     return variants, target_id

