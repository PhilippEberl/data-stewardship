import sklearn.metrics
from sklearn.model_selection import train_test_split
from tpot import TPOTRegressor
import matplotlib.pyplot as plt
import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import LinearSVR
from sklearn.neighbors import KNeighborsRegressor

tpot_config_DecTree = {
    'sklearn.tree.DecisionTreeRegressor': {
        'max_depth': range(1, 10),
        'min_samples_split': range(2, 10),
        'min_samples_leaf': range(1, 10)
    }
}

tpot_config_linearSVR = {
    'sklearn.svm.LinearSVR': {
        'loss': ["epsilon_insensitive", "squared_epsilon_insensitive"],
        'dual': [True, False],
        'tol': [1e-5, 1e-3, 1e-1],
        'C': [1e-4, 1e-2, 0.5, 1., 10., 20.],
        'epsilon': [1e-3, 1e-1, 1.]
    }
}


tpot_config_KNN = {
    'sklearn.neighbors.KNeighborsRegressor': {
        'n_neighbors': range(1, 10),
        'weights': ["uniform", "distance"],
        'p': [1, 2]
    },
}


def eval_negMSE():
    #X = np.loadtxt("../datasets/exercise2/covid/covid-vaccination-vs-death_ratio_input.data", delimiter=";")
    #y = np.loadtxt("../datasets/exercise2/covid/covid-vaccination-vs-death_ratio_classes.data", delimiter=";")

    #X = np.loadtxt("../datasets/exercise2/wine/wine_red_input.data", delimiter=";")
    #y = np.loadtxt("../datasets/exercise2/wine/wine_red_classes.data", delimiter=";")


    #X = np.loadtxt("../datasets/exercise2/wine/wine_white_input.data", delimiter=";")
    #y = np.loadtxt("../datasets/exercise2/wine/wine_white_classes.data", delimiter=";")

    # y is 3 dimensional array, needs to be changed to 1dimensional. just select the first column of y
    X = np.loadtxt('../datasets/exercise2/solarflares/flare_input.data',delimiter=";")
    y = np.loadtxt('../datasets/exercise2/solarflares/flare_classes.data',delimiter=";")[:,0]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

    tpot = TPOTRegressor(generations=5, population_size=20, verbosity=2,
                        config_dict=tpot_config_KNN)

    tpot.fit(X_train, y_train)
    tpot.export(output_file_name='KNN_SolarFlare_pipeline.py')
    print(tpot.fitted_pipeline_)
    print(f"Negative Mean squared error is :{tpot.score(X_test, y_test)}")

def eval_r2():
    X = np.loadtxt("../datasets/exercise2/covid/covid-vaccination-vs-death_ratio_input.data", delimiter=";")
    y = np.loadtxt("../datasets/exercise2/covid/covid-vaccination-vs-death_ratio_classes.data", delimiter=";")

    #X = np.loadtxt("../datasets/exercise2/wine/wine_red_input.data", delimiter=";")
    #y = np.loadtxt("../datasets/exercise2/wine/wine_red_classes.data", delimiter=";")

    #X = np.loadtxt("../datasets/exercise2/wine/wine_white_input.data", delimiter=";")
    #y = np.loadtxt("../datasets/exercise2/wine/wine_white_classes.data", delimiter=";")

    # y is 3 dimensional array, needs to be changed to 1dimensional. just select the first column of y
    #X = np.loadtxt('../datasets/exercise2/solarflares/flare_input.data',delimiter=";")
    #y = np.loadtxt('../datasets/exercise2/solarflares/flare_classes.data',delimiter=";")[:,0]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

    tpot = TPOTRegressor(generations=5, population_size=20, verbosity=2,
                        scoring='r2', config_dict=tpot_config_KNN)

    tpot.fit(X_train, y_train)
    tpot.export(output_file_name='KNN_Covid_pipeline_r2.py')
    print(tpot.fitted_pipeline_)
    print(f"R2 score is :{tpot.score(X_test, y_test)}")

if __name__ == '__main__':
    eval_r2()


