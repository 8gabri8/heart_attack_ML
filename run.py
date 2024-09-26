
# Please set the seed for randomenes

import os
import argparse
import numpy as np
from functions.helpers import *
from functions.implementations import *
from functions.my_functions import *


def main():

    # Set random seed
    SEED = 42
    np.random.seed(SEED)
    
    # Set up argument parser
    parser = argparse.ArgumentParser()
    parser.add_argument("InputDir", type=str, help="Path to the folder containing the dataset.")
    args = parser.parse_args()# Parse the arguments

    DATASET_DIR = args.InputDir

    # Check directory existence
    if not os.path.exists(DATASET_DIR):
        raise FileNotFoundError(f"Error: The directory '{DATASET_DIR}' does not exist.")
    # Check directory non-emptiness
    if not os.listdir(DATASET_DIR):
        raise ValueError(f"Error: The directory '{DATASET_DIR}' is empty.")
    print(f"Success: Dataset directory '{DATASET_DIR}' exists and contains files.")



if __name__ == "__main__":
    main()