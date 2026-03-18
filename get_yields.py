from calculate_autonet_yields import autonetYields
import pickle
import os

autonet_dir = "AutoNets6_Batch_NP"
autonet_data = "AutoNets6_Batch.pkl"
autonet_path = f"{autonet_dir}/{autonet_data}"

num_workers = 32

E_yields, B_yields, viability = autonetYields(autonet_path, num_workers=num_workers)

with open(f"{autonet_dir}/Yields_{autonet_data}", "wb") as f:
    pickle.dump((E_yields, B_yields, viability), f)