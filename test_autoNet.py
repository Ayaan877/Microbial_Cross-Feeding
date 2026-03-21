'''
MINIMAL PATHWAY COMBINATION TEST
Returns: Minimal autonomous network (and saves it)
'''
from load_paths import loadPaths
from combine_pathways import buildAutonomousNetwork
from calculate_yield import *
from load_data import *
import time

if __name__ == "__main__":
    mode = "batch"
    all_paths, data_dir = loadPaths(mode=mode, dataset=6)
    paths = [all_paths[i][3] for i in range(8)] # Take 1st pathway for each target

    start = time.time()
    MinNet = buildAutonomousNetwork(paths, rxnMat, prodMat, sumRxnVec, 
                                    nutrientSet, Currency, Core, prune=True, verbose=True)
    print(f"Time taken: {(time.time() - start)} seconds")

    with open('MinNet_test.pkl', "wb") as f:
            pickle.dump(MinNet, f)
