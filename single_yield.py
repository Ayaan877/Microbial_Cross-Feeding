"""
single_yield.py

Calculate and print stoichiometric biomass and energy yields for a single
autonomous network or cross-feeding pair.

Configuration
-------------
MODE      : "auto" | "cross"
NET_FILE  : path to the .pkl file containing the network list
NET_INDEX : index of the network / pair to inspect
"""

import pickle
import numpy as np

from get_stoich_yields import biomass_cost, fit_cost
from load_data import *

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

MODE      = "cross"                                          # "auto" | "cross"
NET_FILE  = "data/networks/crossnets_mp_batch_NP_pv2_v1_byp_v1.pkl"           # path to network file
NET_INDEX = 73                                               # which network to inspect

# ---------------------------------------------------------------------------

def _rxn_names(rxns):
    return [inv_rxn_map.get(int(r), str(r)) for r in rxns]

def _met_name(idx):
    return inv_met_map.get(int(idx), str(idx))


def print_auto_yields(net, index):
    rxns = np.asarray(net, dtype=int)
    bms  = biomass_cost(rxns)
    fit  = fit_cost(rxns)

    print(f"\n=== Autonomous Network (index {index}) ===")
    print(f"  Autonomous Reactions : {len(rxns)}")
    print(f"  Biomass yield : {bms:.6f}")
    print(f"  Energy yield  : {fit:.6f}")


def print_cross_yields(pair, index):
    cross_A   = np.asarray(pair['cross_A'],  dtype=int)
    cross_B   = np.asarray(pair['cross_B'],  dtype=int)
    B_donated = int(pair['B_donated'])
    A_donated = int(pair['A_donated'])

    nut_A = list(set(nutrientSet) | {B_donated})
    nut_B = list(set(nutrientSet) | {A_donated})

    bms_A = biomass_cost(cross_A, nut_A)
    fit_A = fit_cost(cross_A, nut_A)
    bms_B = biomass_cost(cross_B, nut_B)
    fit_B = fit_cost(cross_B, nut_B)

    print(f"\n=== Cross-Feeding Pair (index {index}) ===")
    print(f"  Organism A reactions : {len(cross_A)}")
    print(f"  Organism B reactions : {len(cross_B)}")
    print(f"  A receives from B : {_met_name(B_donated)}")
    print(f"  B receives from A : {_met_name(A_donated)}")
    print(f"  Organism A — biomass: {bms_A:.6f}  |  energy: {fit_A:.6f}")
    print(f"  Organism B — biomass: {bms_B:.6f}  |  energy: {fit_B:.6f}")


if __name__ == "__main__":
    import time

    start = time.time()
    print(f"Loading '{NET_FILE}' ...")
    with open(NET_FILE, "rb") as f:
        networks = pickle.load(f)
    print(f"  {len(networks)} entries loaded")

    if NET_INDEX >= len(networks):
        raise IndexError(
            f"NET_INDEX={NET_INDEX} is out of range for {len(networks)} entries."
        )

    entry = networks[NET_INDEX]

    if MODE == "auto":
        print_auto_yields(entry, NET_INDEX)
    elif MODE == "cross":
        print_cross_yields(entry, NET_INDEX)
    else:
        raise ValueError(f"Unknown MODE '{MODE}'. Use 'auto' or 'cross'.")

    end = time.time()
    print(f"Elapsed time: {end - start:.2f} seconds")
