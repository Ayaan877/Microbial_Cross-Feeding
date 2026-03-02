'''
REVERSE SCOPE AND PRUNING TEST
Returns: Minimal pathways for a given precursor
'''
from reverse_scope import giveRevScope
from batch_pruning import randMinNetwork  
# Replace 'batch_pruning' with 'single_pruning'/'simple_single_pruning' to test other methods
from load_data import *

if __name__ == "__main__":

    with open("inv_rxn_map.pkl", "rb") as f:
        inv_rxn_map = pickle.load(f)

    target = Core[0] # Test on C00022 (Pyruvate)
    target_id = inv_met_map[target]
    target_name = cpd_string_dict[target_id]
    print(f"Testing pruning on target: {target_id} - {target_name}")
    print("Running reverse scope...")
    
    satMets, satRxns = giveRevScope(rxnMat, prodMat, sumRxnVec, nutrientSet, Currency, target)
    print(f"Reverse scope complete, with {np.sum(satRxns)} satisfied reactions. Starting pruning...")

    RS_rxns = randMinNetwork(satRxns, rxnMat, prodMat, sumRxnVec, target, 
                             nutrientSet, Currency, rng=np.random.default_rng(42))
    rxn_ids = [inv_rxn_map[idx] for idx in RS_rxns]

    print("\nPrecursor Name:", target_id, target_name)
    print("Satisfied Metabolites:", np.sum(satMets))
    print("Satisfied Reactions:", np.sum(satRxns))
    print("Minimal Reactions:", rxn_ids)