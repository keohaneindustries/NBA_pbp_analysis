# %% imports
import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.linear_model import LogisticRegression

from nba_pbp_analysis.data.local_io_utils import LocalIOUtils
from project_data import get_stored_classifier_data, convert_stored_classifier_data_to_predictors, \
    drop_extraneous_predictors

# %% globals

SAVE_CORR_COEF_TO_CSV = True


# %% fx

def calc_corr_coef(df: pd.DataFrame, save_to_csv: bool = False):
    df_corr = df.corr()
    print("correlation matrix:\n{}".format(df_corr.applymap('{:,.2f}'.format)))
    if save_to_csv is True:
        LocalIOUtils.save_csv(df=df_corr.round(4), fileid="corr_matrix_coef")
    return df_corr


def _log_reg_sklearn(X_train, X_test, y_train, y_test):
    _start_time = time.time()
    print("\nfitting Logistic Regression (sklearn)...")
    # Running Model #
    lr = LogisticRegression()
    lr.fit(X_train, y_train)
    _end_time = time.time()
    print("fitted Logistic Regression (sklearn) in {:0.1f} seconds.".format(_end_time - _start_time))

    # Stats #
    score = lr.score(X_test, y_test)
    conf_matrix = confusion_matrix(y_test, lr.predict(X_test))
    cr = classification_report(y_test, lr.predict(X_test))

    print(f'model accuracy (logistic regression, sklearn) = {round(score, 2)}')
    print("confusion matrix (logistic regression, sklearn):\n{}".format(conf_matrix))
    print("classification report (logistic regression, sklearn):\n{}".format(cr))
    return


def _log_reg_statsmodels(X_train, X_test, y_train, y_test, add_constant: bool = True):
    if add_constant is True:
        X_train = sm.add_constant(X_train)
        X_test = sm.add_constant(X_test)

    _start_time = time.time()
    print("\nfitting Logistic Regression (statsmodels)...")
    # Running Model #
    lr = sm.Logit(y_train, X_train).fit()
    _end_time = time.time()
    print("fitted Logistic Regression (statsmodels) in {:0.1f} seconds.".format(_end_time - _start_time))

    # printing the summary table
    print(lr.summary())

    # performing predictions on the test datdaset
    y_pred_cont = lr.predict(X_test)
    y_pred_binary = y_pred_cont >= 0.5

    conf_matrix = confusion_matrix(y_test, y_pred_binary)
    cr = classification_report(y_test, y_pred_binary)
    print("confusion matrix (logistic regression, statsmodels, binary):\n{}".format(conf_matrix))
    print("classification report (logistic regression, statsmodels, binary):\n{}".format(cr))

    return


## Function for Question 2 - Logistic Regression ##
def log_reg(df, test_size=0.2, seed=2):
    # Setting Seed #
    np.random.seed(seed)

    # Extracting Features #
    X = df.iloc[:, 1:]
    y = df.iloc[:, 0]

    X_train, X_test, y_train, y_test = \
        train_test_split(X, y, test_size=test_size, random_state=42)

    _log_reg_sklearn(X_train, X_test, y_train, y_test)
    _log_reg_statsmodels(X_train, X_test, y_train, y_test)


# %% main
def main():
    df_classifier_data = get_stored_classifier_data()
    df_classifier_data = drop_extraneous_predictors(convert_stored_classifier_data_to_predictors(df=df_classifier_data))
    _ = calc_corr_coef(df=df_classifier_data, save_to_csv=SAVE_CORR_COEF_TO_CSV)
    log_reg(df=df_classifier_data)
    return


if __name__ == '__main__':
    main()
