from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeRegressor

import random as rd
import numpy as np


def get_mutation_value(value, digit, base):
    return (value % base**(digit+1)) // base**digit


def init_arguments(min_max_values, integer_values=None):
    arguments = []

    for i, mm in enumerate(min_max_values):
        if not integer_values or integer_values[i]:
            arguments.append(rd.randint(mm[0], mm[1]))
        else:
            arguments.append(rd.uniform(mm[0], mm[1]))

    return arguments


def mutate_arguments_direct_neightborhood(arguments, changes, min_max_values, include_original=True):
    changes_per_value = len(changes[0])
    total_changes = len(changes)**changes_per_value

    mutated_arguments = []

    for i in range(len(arguments)):
        for j in range(len(changes[i])):
            new_arguments = arguments.copy()
            new_arguments[i] += changes[i][j]
            new_arguments[i] = max(min_max_values[i][0], min(
                new_arguments[i], min_max_values[i][1]))
            if include_original or new_arguments != arguments:
                mutated_arguments.append(new_arguments)

    return mutated_arguments


def get_random_direct_mutation(arguments, changes, min_max_values):
    i = rd.randint(0, len(arguments)-1)
    j = rd.randint(0, len(changes[i])-1)

    new_arguments = arguments.copy()
    new_arguments[i] += changes[i][j]
    new_arguments[i] = max(min_max_values[i][0], min(
        new_arguments[i], min_max_values[i][1]))

    return new_arguments


def mutate_arguments_full_neighborhood(arguments, changes, min_max_values, include_original=True):
    changes_per_value = len(changes[0])
    total_changes = len(changes)**changes_per_value

    mutated_arguments = []

    for i in range((0 if include_original else 1), total_changes):
        new_arguments = arguments.copy()

        for j in range(len(new_arguments)):
            new_arguments[j] += changes[j][get_mutation_value(
                i, j, changes_per_value)]
            new_arguments[j] = max(min_max_values[j][0], min(
                new_arguments[j], min_max_values[j][1]))

        mutated_arguments.append(new_arguments)

    return mutated_arguments


def get_random_full_mutation(arguments, changes, min_max_values):
    changes_per_value = len(changes[0])
    total_changes = len(changes)**changes_per_value

    mutated_arguments = []

    i = rd.randint(0, total_changes)

    mutated_arguments = arguments.copy()

    for j in range(len(mutated_arguments)):
        mutated_arguments[j] += changes[j][get_mutation_value(
            i, j, changes_per_value)]
        mutated_arguments[j] = max(min_max_values[j][0], min(
            mutated_arguments[j], min_max_values[j][1]))

    return mutated_arguments


def calculate_quality(arguments, X, y):
    # change with evaluation of regression with given arguments
    # example:
    criterion = ["friedman_mse", "poisson"]
    splitter = ["best", "random"]
    regressor = DecisionTreeRegressor(
        criterion=criterion[arguments[0]],
        splitter=splitter[arguments[1]],
        max_depth=arguments[2],
        min_samples_split=arguments[3],
        min_samples_leaf=arguments[4],
        max_features=arguments[5]
    )

    #X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)
    #regressor.fit(X, y)
    #pred = regressor.predict(X)
    # for i in range(len(X)):
    #	print(pred[i], y[i])

    cvs = cross_val_score(regressor, X, y, cv=10)
    return sum(cvs) / len(cvs)


def evaluate(mutated_arguments, X, y):
    evaluated_arguments = []

    for arguments in mutated_arguments:
        evaluated_arguments.append(
            [calculate_quality(arguments, X, y), arguments])

    evaluated_arguments.sort(reverse=True)

    return evaluated_arguments


def hill_climbing(X, y, changes, min_max_values, integer_values=None):
    best_arguments = [12]  # init_arguments(min_max_values, integer_values)
    best_score = evaluate([best_arguments], X, y)[0][0]

    no_change_since = 0
    max_no_change_since = 1

    print([best_score, best_arguments])

    while(no_change_since < max_no_change_since):
        mutated_arguments = mutate_arguments_direct_neightborhood(
            best_arguments, changes, min_max_values)
        evaluated_arguments = evaluate(mutated_arguments, X, y)

        if evaluated_arguments[0][0] > best_score:
            best_score = evaluated_arguments[0][0]
            best_arguments = evaluated_arguments[0][1]
        else:
            no_change_since += 1

        print(evaluated_arguments[0])

    return best_score, best_arguments


def simmulated_annealing(X, y, changes, min_max_values, integer_values=None):
    best_arguments = init_arguments(min_max_values, integer_values)
    best_score = evaluate([best_arguments], X, y)[0][0]
    current_arguments = best_arguments
    current_score = best_score

    start_arguments = best_arguments

    no_change_since = 0
    max_no_change_since = 10

    generations_to_kill = 1000

    temperature = 1000

    print([best_score, best_arguments])

    while(no_change_since < max_no_change_since and generations_to_kill > 0):
        mutated_argument = get_random_full_mutation(
            current_arguments, changes, min_max_values)
        evaluated_argument = evaluate([mutated_argument], X, y)[0]

        print(evaluated_argument)

        if not np.log(np.random.rand()) * temperature > (evaluated_argument[0] - current_score):
            current_score = evaluated_argument[0]
            current_arguments = evaluated_argument[1]
        elif temperature < 1:
            no_change_since += 1

        if current_score >= best_score:
            best_score = current_score
            best_arguments = current_arguments

        temperature = 0.9 * temperature
        generations_to_kill -= 1

    print("start arguments", start_arguments)
    print("best arguments", best_arguments)

    return best_score, best_arguments


def main():
    X = np.loadtxt(
        '../datasets/exercise2/solarflares/flare_input.data', delimiter=";")
    y = np.loadtxt(
        '../datasets/exercise2/solarflares/flare_classes.data', delimiter=";")

    # criterion, splitter, max_depth, min_samples_split, min_samples_leaf, max_features
    min_max_values = [[0, 1], [0, 1], [1, 50],
                      [0.01, 0.5], [0.01, 0.5], [0, 1]]
    integer_values = [True, True, True, False, False, False]
    changes = [[1, 0, -1], [1, 0, -1], [2, 0, -2],
               [0.01, 0, -0.01], [0.01, 0, -0.01], [0.01, 0, -0.01]]

    best_score, best_arguments = simmulated_annealing(
        X, y, changes, min_max_values, integer_values=integer_values)

    print("best score: {}".format(best_score))
    print("best arguments: {}".format(best_arguments))


if __name__ == "__main__":
    main()
