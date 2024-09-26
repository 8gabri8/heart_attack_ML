# heart_attack_ML

The documentation must also include all data preparation, feature generation as well as cross-validation steps that you have used

PARSER FOR THE INPUT FOLDER WTH DATA (PUT ALSO IN THE DOCUMENTATION)
TRY IMPLEMENT SVM

ATTEMTION IN OGISTC REGRESSOON LABELS MUST BE (0.1)

# Column/feature selection
We wanted to reduce the number of features to avoid curse od dimentisonlity

We decided to use cony "calculted vriables" because
    - they ar ea few
    - they summurce amny other vairables
    - they are considered good indicatior of HT by the clinciacians/the guy wh made the sudy
        cite"s. Thecommon focus of these variables is on health behaviors that are associated with a risk of illness or injury" https://www.cdc.gov/brfss/annual_data/2015/pdf/2015_Calculated_Variables_Version4_08_10_17-508c.pdf

We decided to also add cols:
    -

We decied to remove the cols with more than % of NaN, as they will not be useful for classification

We removed cols with high correlation
We mantined columns highly corrected with response variable

IN CONTINOUS VARIBALES CHNAGE THE VALUE OF THE "MISSING" CATEGORY (9999000 CAN SCREW THE NORMALIZATION )

.
## Exploratory data analysis 
- only use column with >thr of NaN
- explore column 
    - what columsn means (https://www.cdc.gov/brfss/annual_data/2015/pdf/codebook15_llcp.pdf)
    - other EDA steps (gabri)

- feautre selection
    - retain only col that make sense for classification
    - impute the missing values
        - decide a startegy
    - correlation analaysis
        - remove cols that higly correlated
    - 
- preporcessing
    - normalization ONLY on non-normalization
    - ?) remove datapoitn so that we have balanced classes
    - feture selection algo

fearture enegerirng
    - PCA
    - combinaing features ex. polynomial



# heck fucntions with the test of proposed by the TAs

# Train
- differt inital guesses of w
- differt gamma for GD
- use differnt optimizer Adam, momentum ...
- differnt normlaization technies
- differt subsets of features
- try use w0 (columns with all 1s)
- use other losses
- use other models: RF
- logstic regression with sthichstic gd
- logistic regression with newtorn method

# testing locally on validation ONOTEHR is test on aicrowd

# Plots
- how the loss decrease in GD over the iterations
- Try to find out which datapoints are wrongly classified and, if possible, why
this is the case.
# link to latex report

# compettite evaluation
[link](https://www.aicrowd.com/challenges/epfl-machine-learning-project-1)
