'''
TOY NETWORK YIELD TEST
Returns: Energy/Biomass yields for a network
'''
from calculate_yield import *
from load_data import * # load 'toynet.csv' in load_data

AutoNet1 = [0, 1, 6] # Uses R1, R2 & R7
AutoNet2 = [2, 3, 4, 5, 7] # Uses R3, R4, R5, R6 & R8
nutrientSet = [0] # X
Energy = []
Currency = []
Core = [6, 7] # BA_star & AA_star

if __name__ == "__main__":
    E1, B1, stat1 = splitByDemand(stoich_matrix, rxnMat, prodMat, 
                                  sumRxnVec, rho, pi, nutrientSet, Energy,
                                  Currency, Core, AutoNet1)
    E2, B2, stat2 = splitByDemand(stoich_matrix, rxnMat, prodMat, 
                                  sumRxnVec, rho, pi, nutrientSet, Energy, 
                                  Currency, Core, AutoNet2)
    
    print(f'Viablity: {stat1*stat2}')
    print(f'Energy Yields:\n    AutoNet 1 = {E1}\n  AutoNet 2 = {E2}')
    print(f'Biomass Yield:\n    AutoNet 1 = {B1}\n  AutoNet 2 = {B2}')
