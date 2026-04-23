'''
NUMPATHS AUTONOMOUS NETWORK CONSTRUCTION TEST
Loads a NumPaths dataset, takes one pathway per target, and builds a
single autonomous network as a quick sanity check.
'''
from load_numpaths import loadNumPaths
from combine_pathways import buildAutonomousNetwork
from load_data import *
import pickle
import time

if __name__ == "__main__":
    mode = "batch"
    all_paths, data_dir = loadNumPaths(mode=mode, dataset=3)
    paths = [all_paths[i][0] for i in range(8)]  # Take 1st pathway for each target

    start = time.time()
    MinNet = buildAutonomousNetwork(paths, rxnMat, prodMat, sumRxnVec,
                                    nutrientSet, Currency, Core, prune=True, verbose=True)
    print(f"Time taken: {(time.time() - start)} seconds")

    with open('AutoNet_test.pkl', "wb") as f:
            pickle.dump(MinNet, f)
