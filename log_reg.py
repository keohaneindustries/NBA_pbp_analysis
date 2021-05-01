# %% imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.linear_model import LogisticRegression

from project_data import get_stored_classifier_data


# %% fx
## Function for Question 2 - Logistic Regression ##
def log_reg(df, test_size=0.2, seed=2):
    # Setting Seed #
    np.random.seed(seed)

    # Extracting Features #
    X = df.iloc[:, 1:]
    y = df.iloc[:, 0]

    X_train, X_test, y_train, y_test = \
        train_test_split(X, y, test_size=test_size, random_state=42)

    # Running Model #
    lr = LogisticRegression()
    lr.fit(X_train, y_train)

    # Stats #
    score = lr.score(X_test, y_test)
    conf_matrix = confusion_matrix(y_test, lr.predict(X_test))
    cr = classification_report(y_test, lr.predict(X_test))

    print(f'The accuracy for the logistic regression model is {round(score, 2)}')
    print()
    print(f'The logistic regression model confusion matrix:')
    print()
    print(conf_matrix)
    print(f'The logistic regression model classification report:')
    print()
    print(cr)


# %% main
def main():
    df_classifier_data = get_stored_classifier_data()
    log_reg(df=df_classifier_data)
    return


if __name__ == '__main__':
    main()
