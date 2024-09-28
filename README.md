# CHECK FUNTISON WITH THERI CODE
# ENROLL IN GITCLASSROOM
# WRITE READ ME
PARSER FOR THE INPUT FOLDER WTH DATA (PUT ALSO IN THE DOCUMENTATION)
# 

# IDEAS
- TRY IMPLEMENT SVM
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
- try differt inital_w

# ATTENTION
ATTEMTION IN OGISTC REGRESSOON LABELS MUST BE (0.1)

# QUESTIONS
- how to solve class imbalance, cut some data?
    - If your dataset contains many more negative samples than positive ones, a model that always predicts the positive class could achieve a high F1 score (for the positive class) while having low accuracy, because it's misclassifying most of the negative samples.
- what put in the final report, the crossvalidation metrics on x_train, or the results of the challenge
- can we roemeve rows from the daatset



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


# testing locally on validation ONOTEHR is test on aicrowd

# Plots
- how the loss decrease in GD over the iterations
- Try to find out which datapoints are wrongly classified and, if possible, why
this is the case.
# link to latex report

# compettite evaluation
[link](https://www.aicrowd.com/challenges/epfl-machine-learning-project-1)
