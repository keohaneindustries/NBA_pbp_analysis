# %% imports
import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt
# from matplotlib.colors import ListedColormap
from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import StandardScaler
# from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix, classification_report
# from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score

from sklearn.neighbors import KNeighborsClassifier
# from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
# from sklearn.decomposition import PCA

from project_data import get_stored_classifier_data, convert_stored_classifier_data_to_predictors, \
    drop_extraneous_predictors


# %% fx
def comparing_models(df, test_size=0.2):
    # Extracting Features #
    X = df.iloc[:, 1:]
    y = df.iloc[:, 0]

    X_train, X_test, y_train, y_test = \
        train_test_split(X, y, test_size=test_size, random_state=42)

    # Setting up the Models #
    classifiers = {
        "Logistic Regression": LogisticRegression(),
        "K Nearest Neighbors": KNeighborsClassifier(),
        "SVM": SVC(kernel='linear'),
        "KSVM": SVC(kernel='rbf'),
        "Neural Network": MLPClassifier(hidden_layer_sizes=(20, 10))
    }

    # Creating a Dictionary for the Accuracies #
    accuracies = {}

    # Running the Process for the Three Classifiers #
    for name, clf in classifiers.items():
        _start_time = time.time()
        print("fitting {}...".format(name))
        clf.fit(X_train, y_train)
        _end_time = time.time()
        print("fitted {} in {:0.1f} seconds.".format(name, _end_time - _start_time))
        score = clf.score(X_test, y_test)
        conf_matrix = confusion_matrix(y_test, clf.predict(X_test))
        cr = classification_report(y_test, clf.predict(X_test))

        accuracies[name] = score

        ## Printing the Accuracies and the Confusion Matrix for Each of the Classifiers ##
        print(f'The accuracy for the {name} model is {round(score, 2)}')
        print()
        print(f'The {name} model confusion matrix:')
        print()
        print(conf_matrix)
        print(f'The {name} model classification report:')
        print()
        print(cr)

    # Plotting the Accuracies #
    plt.figure(figsize=(10, 5))
    plt.bar(accuracies.keys(), accuracies.values())
    plt.ylim((0., 1.))

    for i, v in enumerate(accuracies.values()):
        plt.text(i - 0.08, v - 0.05, "{:0.2%}".format(v), color='w')

    plt.title("Accuracy by Model\n(n_train={:,}; n_test={:,})".format(len(X_train), len(X_test)))
    plt.tight_layout()
    plt.savefig('images/model_comparison_accuracy.png')
    plt.show()


# %% main
def main():
    df_classifier_data = get_stored_classifier_data()
    df_classifier_data = drop_extraneous_predictors(convert_stored_classifier_data_to_predictors(df=df_classifier_data))
    comparing_models(df_classifier_data)
    return


if __name__ == '__main__':
    main()
