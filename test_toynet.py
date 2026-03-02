'''
TOY NETWORK YIELD TEST
Returns: Energy/Biomass yields for a network
'''
from calculate_yield import *
from load_data import * # load 'toynet.csv' in load_data

AutoNet1 = [0, 1, 6] # Uses R1, R2 & R7
AutoNet2 = [2, 3, 4, 5, 7] # Uses R3, R4, R5, R6 & R8
CrossNet = [0, 1, 2, 3, 4, 5, 6, 7] # Crossfeeding uses all
nutrientSet = [0] # X
Energy = []
Currency = []
Core = [6, 7] # BA_star & AA_star

if __name__ == "__main__":
    E_yield, B_yield, status = splitByDemand(stoich_matrix, rxnMat, prodMat, 
                                          sumRxnVec, rho, pi, nutrientSet, Energy, 
                                          Currency, Core, AutoNet1)
    
    print(f'All precursors produced: {status}')
    print(f'Energy Yield: {E_yield}')
    print(f'Biomass Yield: {B_yield}')
