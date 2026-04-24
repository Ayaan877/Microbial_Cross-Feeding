'''
MINPATHS AUTONOMOUS NETWORK CONSTRUCTION TEST
Loads a MinPaths dataset, takes one pathway per target, and builds a
single autonomous network as a quick sanity check.
'''
from load_minPaths import loadMinPaths
from combine_pathways import buildAutonomousNetwork
from load_data import *
import pickle
import time

if __name__ == "__main__":
    mode = "batch"
    all_paths = loadMinPaths(mode=mode, dataset=2)
    paths = [all_paths[i][0] for i in range(8)]

    start = time.time()
    MinNet = buildAutonomousNetwork(paths, rxnMat, prodMat, sumRxnVec,
                                    nutrientSet, Currency, Core, prune=True, verbose=True)
    print(f"Time taken: {(time.time() - start)} seconds")

    with open('autoNet_test.pkl', "wb") as f:
            pickle.dump(MinNet, f)
