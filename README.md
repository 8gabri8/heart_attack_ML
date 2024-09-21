# heart_attack_ML

The documentation must also include all data preparation, feature generation as well as cross-validation steps that you have used
.
## Exploratory data analysis 
You should learn about your dataset - figure out which features are continuous, which
ones are categorical, check if there are obvious relationships between the features, take a look at the distribution
of each feature, and so on.

## Preprocessing
- normalize/scaling
- filter feature that are not Nan for at least N% of the observation
- impute NaN values
- remove correlated feautres
- A feature may be important if it is highly correlated with the dependent variable (the thing being predicted).
- combinaing features ex. polynomial
- PCA

# Train
- differt inital guesses of w
- differt gamma for GD
- use differnt optimizer Adam, momentum ...
- use other losses

# Plots
- how the loss decrease in GD over the iterations
