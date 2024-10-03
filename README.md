cheu"k" fu(cantonise) --> big trustable
# NEXT WEEK sunday 6 oct (gabri zoom)
    - SOME RESULTS NAD CHECK THE CODE
        LOGIST (zhuofu)
        LINEAR AND LEAST SQUARE (gabri)
        SVM
    - PLOTS (sofiia) --> LOOK DONW!!!!
    - TESTS FUNcTION (zhuofo)
    - TRY COMPETITION (gabri)
    - LATEX (sofiia)
    - QUESTION TAs (sofiia, zhuofu)
    - MICHALE NEURO mail (gabri)
ì

# CHECK FUNTISON WITH THERI CODE
# ENROLL IN GITCLASSROOM
# TRY THE COMPETITION
# LATEX REPORT
# WRITE READ ME
PARSER FOR THE INPUT FOLDER WTH DATA (PUT ALSO IN THE DOCUMENTATION)

# PLOTS (sofiia)
- PUT THE ROC CRUVE PLOT
- CORRELATION VAR 
- VIF VAIBLES
- hist  (before inputing, after imputing and withou nan_class)
- SCATTERPLOTS for continuops
- other ideas
- class invarince

# ATTENTION
ATTEMTION IN OGISTC REGRESSOON LABELS MUST BE (0.1)

# QUESTIONS FOR TAs
- how to solve class imbalance, cut some data?
answer: he is not allowed to answer) it is the first main challange for the project to solve  he recommended to try some regularisatoin or aother approaches)
    - shoould I train on a subsample of the data?
	answer: try)
    - If your dataset contains many more negative samples than positive ones, a model that always predicts the positive class could achieve a high F1 score (for the positive class) while having low accuracy, because it's misclassifying most of the negative samples.
- what put in the final report, the crossvalidation metrics on x_train, or the results of the challenge
answer: both
- can we roemeve rows from the daatset
answer: try)
- does the run.py shoduld run only the best run that we have created or give the possibilty fo runnign everything we have done??
answer: only final
- the graphs/plots shoduld be created from the run file? also the oen of EDA?
answer: he said it should be fine. i would ask in the forum
- can w3e use the library Json to load and upload file? 
answer: ask in the edforum
- how to chosse the best model in corss validation? the one with bigger accuracy or F1? how?
answer: check the theory, should be f1
- what means if I have big labdas (10^8) in ridge is this possbile? 
answer: you probably really penelise the model (or opposite), you should try different parameters.


# compettite evaluation
[link](https://www.aicrowd.com/challenges/epfl-machine-learning-project-1)


# FEATURES
1) select only "suggested by physicians"
2) select only continous, as discrete are just discretization of the continous
3) correlation analysis	(#high correlation with rest)	
	'FC60_', 'MAXVO2_', 'PAMIN11_', 'PAMIN21_' 
	'FRUTDA1_', #Fruit intake in times per day
	'GRENDAY_', #Dark green vegetable intake in times per day
	'VEGEDA1_', #Other vegetable intake in times per day 
	'METVL11_', # cannot underatnd
	'METVL21_', 
	'PA1VIGM_', #Minutes of total Vigorous Physical Activity per week
4) manual check of the remaing variables (ex. very simoalr to each to each others)
	'PADUR1_', #Minutes of First Activity
	'PADUR2_', #Minutes of Second Activity 
	'PAFREQ1_', #Physical Activity Frequency per Week for First Activity
	'PAFREQ2_', #Physical Activity Frequency per Week for Second Activit
	'PAVIG11_', #Minutes of Vigorous Physical Activity per week for First Activity
	'PAVIG21_', #Minutes of Vigorous Physical Activity per week for Second Activity

5) reimaing features
[
'_AGE80', #Imputed Age value collapsed above 80
'_BMI5', #Body Mass Index (BMI)
'_DRNKWEK' #Calculated total number of alcoholic beverages consumed per week
'DROCDY3_', #Drink-occasions-per-day
'_FRUTSUM', #Total fruits consumed per day
'_MINAC11', #Minutes of Physical Activity per week for First Activity
'_MINAC21', #Minutes of Physical Activity per week for Second Activity
'_VEGESUM', #Total vegetables consumed per day
'ORNGDAY_', #Orange-colored vegetable intake in times per day
'PA1MIN_', #Minutes of total Physical Activity per week
]
6) * AMYBE FILTER MORE --> SEE REUSLT OF MODELS
7) plot their distribution
8) imputr values
8) feaure engeriign
    - subsample feaures
    - pca
    - polynotmila degree





*) good discrete features that we can try:
"_PNEUMO2", "_RACEGR3", "_RFBING5", "_RFHYPE5", "_SMOKER3", "_ASTHMS1", "_CHOLCHK", "_DRDXAR1", "_EDUCAG", "_FRT16"
