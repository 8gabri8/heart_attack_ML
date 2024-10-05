
# Please set the seed for randomenes

import os
import argparse
import json
import numpy as np
from functions.helpers import *
from functions.implementations import *
from functions.implementations_helper import *
from functions.training_functions import *


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

    #################################
    # Configuration Parameters
    #################################

    ONLY_FEATURE_TO_USE = [""]  
    SUGGESTED_FEATURE_PATH = "data/suggested_features_dict.json"  # Path to the JSON file containing suggested features
    NORMALIZATION_METHOD = "z-score"  # Method to normalize continuous features during preprocessing
    FEATURE_ENGENEERING_TYPE = 1  # Type of feature assembly during feature engineering (see process_data function for details)
    THR_VAR_PCA = 0.9  # Percentage of variance to retain during PCA
    MAX_NAN_PER_COL = 0.75  # Max percentage of NaNs in one column before removal
    PREDICTION_METHOD = "least_squares"  # Prediction method (see training_function.py for options)
    LABDA_REGULARIZATION = 0.1  # Regularization parameter for Lasso or Ridge regression
    GAMMA_OPTIMIZER = 0.01  # Learning rate during optimization methods
    INITIAL_W = "zeros"  # Method to initialize weights in GD or logistic regression (see training_function.py for options)
    MAX_ITERS = 100 #  max iters in optimization method
    SPECIFIC_FEATURES = ["_AGE80", "_BMI5"], # Subset of feauture to use, ignoring all the others (in case is choosen specfiic type of feature engeneering)
    DEGREE_POL = 3 # Max degree of feature polinomial expansion (in case is choosen specfiic type of feature engeneering)
    K_FOLDS = 10 # Number of folds in stratified corss validation
    UNCORRELATED_FEATURES = [
                        "_PNEUMO2", "_RACEGR3", "_RFBING5", "_RFHYPE5", "_SMOKER3", 
                        "_ASTHMS1", "_CHOLCHK", "_DRDXAR1", "_EDUCAG", "_FRT16",#continous

                        #'DROCDY3_', 'FRUTDA1_', '_FRUTSUM', '_VEGESUM', 'GRENDAY_', 
                        #'METVL11_', 'METVL21_', 'ORNGDAY_', 'PA1MIN_', 'PA1VIGM_', 
                        #'PADUR1_', 'PADUR2_', 'PAFREQ1_', 'PAFREQ2_', 'PAMIN11_', 
                        #'PAMIN21_', 'PAVIG11_', 'PAVIG21_', 'VEGEDA1_', '_AGE80', 
                        #'_BMI5', '_DRNKWEK' #discrete
                    ]# with the features suggested by physicinas, subset of uncorrelated ones (see EDA analysis on how they have been choosen)


    #################################
    # Load data
    #################################

    print("Loading dataset (could take few minutes)...")
    x_train, x_test, y_train, train_ids, test_ids = load_csv_data(DATASET_DIR, sub_sample=False)
    print("Dataset Loaded.")

    # Extract the columns name (feature name)
    feature_name = np.genfromtxt(os.path.join(DATASET_DIR, 'x_train.csv'), delimiter=',', dtype=str, max_rows=1)
    feature_name = feature_name[1:] # The "ids" cols have been removed in the load_csv_data() fucntion

    #################################
    # Load Features suggested by physician
    #################################
 
    with open(SUGGESTED_FEATURE_PATH, "r") as f:
        suggested_features_dict = json.load(f)

    #################################
    # Preprocess features training set (x_train)
    #################################

    x_train = process_train_data(
        x_train=x_train,
        feature_name=feature_name,  # Features (column names) of the x_train matrix
        data_dict=suggested_features_dict,     # Dict where keys are feature names, and values are lists: [nan symbol, feature type ('continuous' or 'discrete')]
        thr_nan=MAX_NAN_PER_COL,  # Threshold above which the column will be discarded (default: 75% NaN)
        thr_var_pca = THR_VAR_PCA, #Percentage of varaince to save during pca
        normalization=NORMALIZATION_METHOD,  # Type of normalization for continuous variables
        feature_engeneering_type=FEATURE_ENGENEERING_TYPE,  # Feature engineering option (default: use only continuous variables)
        degree_pol = DEGREE_POL, #Max degree of feature polinomial expansion (in case is choosen specfiic type of feature engeneering)
        uncorrelated_features = UNCORRELATED_FEATURES,
        specific_features=SPECIFIC_FEATURES, #Subset of feauture to use, ignoring all the others
    )

    np.save('data/x_train_discrete_1hot.npy', x_train)


    # Calculate VIF
    #vif_values = calculate_vif(x_train)
    #print("VIF values for each feature:")
    #print(vif_values)

    #################################
    # Train Model
    #################################

    if PREDICTION_METHOD in ["mse_gd", "mae_gd", "ridge_gd", "ridge_adam", "lasso_gd", "lasso_adam", "least_squares"]:
        fold_metrics, \
        best_w, \
        best_accuracy, \
        best_f1_score, \
        best_confusion_matrix, \
        mean_val_accuracy, \
        mean_val_f1_score, \
        mean_train_accuracy, \
        mean_train_f1_score = stratified_k_fold_cross_validation(
            y = y_train,  # Labels for the dataset
            tx = x_train,  # Feature matrix
            k_folds = K_FOLDS,  # Number of folds for cross-validation
            max_iters = MAX_ITERS,  # Maximum iterations for optimization
            gamma = GAMMA_OPTIMIZER,  # Learning rate for optimization
            lambda_regularization = LABDA_REGULARIZATION,  # Regularization parameter
            prediction_method = PREDICTION_METHOD ,  # Method used for optimization
            initialization_w = INITIAL_W,  # Weight initialization method
            seed=SEED,  # Seed for reproducibility
        )

    #################################
    # Save Best Model
    #################################

    #both model
    #both parmater of modle and preprocessing used to train


    #################################
    # Load Best Model
    #################################


    #################################
    # Preprocess features testing set (x_test)
    #################################

    # Same preprocesing done in trainin

    #################################
    # Predict Test Labels
    #################################

    print(x_train)
    print(best_confusion_matrix)

if __name__ == "__main__":
    main()