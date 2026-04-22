import pickle
import time
from calculate_yield import splitByDemand
from calculate_crossfeeding_yield import splitByDemand_crossfeeding
from load_data import *
from autonomy_check import verify_autonomy

autonet_path = 'AutoNets_revScope/AutoNets_revScope_2.pkl'
autonet_idx = 49265

with open(autonet_path, 'rb') as f:
    AutoNets = pickle.load(f)

print(f'Loaded {len(AutoNets)} autonomous networks from {autonet_path}')

autonet = AutoNets[autonet_idx]

viable, missing_cores = verify_autonomy(autonet, rxnMat, prodMat, sumRxnVec, nutrientSet, Currency, Core)
produced_cores = [c for c in Core if c not in missing_cores]

print(f'Autonomy check: {"Viable" if viable else "Not viable"}')
print(f'Missing core metabolites: {missing_cores}')
print(f'Produced core metabolites: {produced_cores}')

t0 = time.time()
E_single, B_single, status_single = splitByDemand(
    stoich_matrix, rxnMat, prodMat, sumRxnVec,
    rho, pi, nutrientSet, Energy, Currency, Core, autonet)
t1 = time.time()

print('\nSingle-network yield test')
print(f'  Network index: {autonet_idx}')
print(f'  Viable: {status_single}')
print(f'  Energy yield: {E_single}')
print(f'  Biomass yield: {B_single}')
print(f'  Time: {t1 - t0:.3f} s')

# Cross-feeding pair yield test
crossnet_path = 'CrossNets_revScope/CrossNets_revScope2_byp2.pkl'
crossnet_idx = 1976

with open(crossnet_path, 'rb') as f:
    CrossNets = pickle.load(f)

print(f'\nLoaded {len(CrossNets)} cross-feeding pairs from {crossnet_path}')

crossPair = CrossNets[crossnet_idx]

viable_A, missing_A = verify_autonomy(crossPair['cross_A'], rxnMat, prodMat, sumRxnVec, nutrientSet, Currency, Core)
viable_B, missing_B = verify_autonomy(crossPair['cross_B'], rxnMat, prodMat, sumRxnVec, nutrientSet, Currency, Core)

print(f'Autonomy check A: {"Viable" if viable_A else "Not viable"} | Missing cores: {missing_A}')
print(f'Autonomy check B: {"Viable" if viable_B else "Not viable"} | Missing cores: {missing_B}')
print(f'Exchanged: A donates met {crossPair["A_donated"]} ({inv_met_map[crossPair["A_donated"]]}), '
      f'B donates met {crossPair["B_donated"]} ({inv_met_map[crossPair["B_donated"]]})')

t2 = time.time()
result = splitByDemand_crossfeeding(
    stoich_matrix, rxnMat, prodMat, sumRxnVec,
    rho, pi, nutrientSet, Energy, Currency, Core, crossPair)
t3 = time.time()

print('\nCross-feeding pair yield test')
print(f'  Pair index: {crossnet_idx}')
print(f'  Network A size: {len(crossPair["cross_A"])} rxns, Network B size: {len(crossPair["cross_B"])} rxns')
print(f'  Pair viable: {result["pair_viable"]}')
print(f'  Network A — viable: {result["viable_A"]}, E: {result["E_A"]}, B: {result["B_A"]}')
print(f'  Network B — viable: {result["viable_B"]}, E: {result["E_B"]}, B: {result["B_B"]}')
print(f'  Flux A→B: {result["flux_A_to_B"]:.6f}, Flux B→A: {result["flux_B_to_A"]:.6f}')
print(f'  Time: {t3 - t2:.3f} s')