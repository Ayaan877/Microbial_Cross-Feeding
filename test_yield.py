import pickle
import time
from calculate_yield import splitByDemand
from load_data import *
from autonomy_check import verify_autonomy

autonet_path = 'AutoNets_revScope/AutoNets_revScope_2.pkl'
autonet_idx = 1925

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