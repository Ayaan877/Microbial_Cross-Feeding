from ast import For
import pickle
import matplotlib.pyplot as plt
import numpy as np

'''
Check Pathways
'''

# dataset = 6
# prune_mode = "Single"
# file = "097"

# with open(f"inv_rxn_map.pkl", "rb") as f:
#     rxn_map = pickle.load(f)

# with open(f"rxn_string_dict.pkl", "rb") as f:
#     rxn_dict = pickle.load(f)

# with open(f"MinNets{dataset}_{prune_mode}/C00{file}_{prune_mode}_MinNets.pkl", "rb") as f:
#     min_paths = pickle.load(f)

# for path in min_paths:
#     # print(f"Pathway with {len(path)} reactions")
#     rxn_ids = []
#     for idx in path:
#         rxn_ids.append(rxn_map[idx])
#         print(rxn_map[idx])
#     print(' ')
#     rxn_strings = []
#     for rxn_id in rxn_ids:
#         print(rxn_dict[rxn_id])
#         rxn_strings.append(rxn_dict[rxn_id])
#     print('--------------------------------------')

'''
Pathway Discovery Rate
'''
dataset = 2
prune_mode = "Batch"
files = ["009", "013", "022", "025", "041", "065", "097", "117"]
fig, axes = plt.subplots(2, 4, figsize=(16, 8))
for ax, file in zip(axes.flat, files):
    with open(f"NumPaths{dataset}_{prune_mode}/C00{file}_Pathways.pkl", "rb") as f:
        results = pickle.load(f)

    unique_counts = np.array(results['unique_counts'])
    attempts = np.array(results['attempts'])*8

    ax.plot(attempts, unique_counts,'o-')
    ax.set_title(f"C00{file}")
    ax.set_xlabel("# of Attempts")
    ax.set_ylabel("Unique Pathways")
    ax.grid()
fig.suptitle(f"Unique Pathway Discovery Rate ({prune_mode})")
plt.tight_layout()
plt.show()

'''
Autonomous Networks and Yields
'''
# dataset = 6
# mode = "Single"
# minimal = "P"

# with open(f"AutoNets{dataset}_{mode}_{minimal}/AutoNets{dataset}_{mode}.pkl", "rb") as f:
#     autonets = pickle.load(f)

# print(f"Number of autonomous networks: {len(autonets)}")

# net_sizes = np.array([len(net) for net in autonets])

# with open(f"AutoNets{dataset}_{mode}_{minimal}/Yields_AutoNets{dataset}_{mode}.pkl", "rb") as f:
#     E_yields, B_yields, viability = pickle.load(f)

# viable_E = E_yields[viability]
# viable_B = B_yields[viability]

# fig, ax = plt.subplots(1, 3, figsize=(12, 4))

# ax[1].hist(viable_E, bins=20, alpha=0.7, color='C0', density=True)
# ax[1].set_xlabel("Energy Yield")
# ax[1].set_ylabel("Fraction of Networks")

# ax[2].hist(viable_B, bins=20, alpha=0.7, color='C2', density=True)
# ax[2].set_xlabel("Biomass Yield")
# ax[2].set_ylabel("Fraction of Networks")

# ax[0].hist(net_sizes, bins=20, alpha=0.7, color='C1', density=True)
# ax[0].set_xlabel("Network Size (# reactions)")
# ax[0].set_ylabel("Fraction of Networks")

# fig.suptitle(f"Efficiency of Viable Autonomous Networks ({mode}-{minimal})")
# plt.tight_layout()
# plt.show()