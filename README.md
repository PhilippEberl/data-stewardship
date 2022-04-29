# The following README describes the experiment in more detail

This experiment compares the performance between AutoML, hillclimbing and simulated annealing. In order to compare the results the Negative Mean Squarred Error (NMSE) and R<sup>2</sup> score are calculated and displayed.

**There is currently an issue open on GitHub of hyperopt-sklearn where the module hpsklearn.estimator is not recognized on import. This is an issue from their side thus all hyperopt notebooks cannot be compiled.**

Github issue: <https://github.com/hyperopt/hyperopt-sklearn/issues/185>

As per the developers, a fix is underway.

## Requirements

Python was used in v3.9 for this experiment.

All notebooks depend on libraries in their most recent version available. They can be found under *requirements.txt* To install all requirements in your virtual environment run

`pip install -r requirements.txt`

---

## How to run each notebook

In order to run each notebook, one can simply open it in VS Code and select at the top of the notebook *Run All*.

---

## Preprocessing

All `.csv`, `.data` and `.name` files for each data set have been preprocessed.

The following preprocessing steps have been applied (see **preprocessing.ipynb**):

* Covid vaccination vs. mortality
  * Conversion to numeric
  * Normalization by dividing through population (e.g. ration of people vaccinated)
  * One hot encoding for country names
* Solar flares
  * Conversion to numeric
  * One hot encoding for region class, largest spot and spot distribution
* Wine quality
  * Conversion to numeric

Used Algorithms (Hillclimbing and simulated anealing) as well as used Regressors (linear SVR, KNN and DecisionTree) will not be explained in this experiment.

---

## Detailed information on data sets

There are three data sets for this experiment. They are provided in the repository so all Jupyter notebooks work without any adaptions. You can also download the raw data from the sources provided below.

### *Covid vaccination vs. mortality*

**Abstract**: This data set shows the ratio between vaccinated people (partially and fully) and new occured deaths.

The COVID data set has the following 10 columns in the `.csv` file

1. id
2. country
3. iso_code
4. date
5. total_vaccinations
6. people_vaccinated
7. people_fully_vaccinated
8. New_deaths
9. population
10. ratio

### *Solar flares*

**Abstract**: Each class attribute counts the number of solar flares of a certain class that occur in a 24 hour period

* The database contains 3 potential classes, one for the number of times a certain type of solar flare occured in a 24 hour period.
* Each instance represents captured features for 1 active region on the sun.
* The data are divided into two sections. The second section (flare.data2) has had much more error correction applied to the it, and has consequently been treated as more reliable.

1. Code for class (modified Zurich class) (A,B,C,D,E,F,H)
2. Code for largest spot size (X,R,S,A,H,K)
3. Code for spot distribution (X,O,I,C)
4. Activity (1 = reduced, 2 = unchanged)
5. Evolution (1 = decay, 2 = no growth, 3 = growth)
6. Previous 24 hour flare activity code (1 = nothing as big as an M1, 2 = one M1, 3 = more activity than one M1)
7. Historically-complex (1 = Yes, 2 = No)
8. Did region become historically complex on this pass across the sun's disk (1 = yes, 2 = no)
9. Area (1 = small, 2 = large)
10. Area of the largest spot (1 = <=5, 2 = >5)

From all these predictors three classes of flares are predicted, which are represented in the last three columns.

11. C-class flares production by this region in the following 24 hours (common flares); Number
12. M-class flares production by this region in the following 24 hours (moderate flares); Number
13. X-class flares production by this region in the following 24 hours (severe flares); Number

### *Wine quality*

**Abstract**: Two datasets are included, related to red and white vinho verde wine samples, from the north of Portugal. The goal is to model wine quality based on physicochemical tests. Both files are available as `.csv`.

Number of Instances: red wine - 1599; white wine - 4898.

Input variables (based on physicochemical tests):

1. fixed acidity
2. volatile acidity
3. citric acid
4. residual sugar
5. chlorides
6. free sulfur dioxide
7. total sulfur dioxide
8. density
9. pH
10. sulphates
11. alcohol

Output variable (based on sensory data):

12. quality (score between 0 and 10)

> ### Sources for data sets
>
> * <https://www.kaggle.com/sinakaraji/covid-vaccination-vs-death/activity>
> * <http://archive.ics.uci.edu/ml/datasets/solar+flare>
> * <https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/>

---

## Metrics used for evaluation

* Negative Mean Squared Error
  * Describes how close a regression line is to a specific set of points. Lower scores indicate a better fit.
* R<sup>2</sup>
  * R-squared explains to what extent the variance of one variable explains the variance of the second variable. Best result is 1.0, which means all of the variation can be explained by the models inputs.

> For more information on the metrics used see:
>
> * <https://scikit-learn.org/stable/modules/generated/sklearn.metrics.mean_squared_error.html>
> * <https://scikit-learn.org/stable/modules/generated/sklearn.metrics.r2_score.html>

---

## Output

### Raw output

Each notebook calculates the negative mean squared error and the R<sup>2</sup> score. They are both displayed for each data set, algorithm and regressor.

### Plots

Furthermore plots should help to visualize the difference in results and help to compare the better. Plots are generated at the very bottom of all notebooks and are not exported.
