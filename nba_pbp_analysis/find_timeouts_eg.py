# %% imports
import datetime as dt
import os
import pandas as pd
import sys
import numpy as np
from datetime import datetime
import time

import sys, argparse, os
sys.path.insert(1,'nba_pbp_analysis/data/')

#from nba_pbp_analysis.data.cf import *
#from nba_pbp_analysis.data.load import load_clean_pbp

# this works because you are currently in the nba
from data.cf import *
from data.load import load_clean_pbp

# %%
def get_two_minute_outcomes(years_requested: list) -> pd.DataFrame:
    _filter_fxs = [
        lambda df: df[df['period'] == 4],
        lambda df: df[
            pd.to_datetime(df['remaining_in_quarter']) <= pd.datetime.today().replace(hour=3, minute=0, second=0)]
    ]
    df = load_clean_pbp(clean_dir=CLEAN_ROOT_DIR, filetype=CLEAN_OUTPUT_FILETYPE, years_requested=years_requested,
                        filter_fxs=_filter_fxs)

    # awaydesc = df['AWAYDESCRIPTION'].dropna()
    # awaytimeouts = awaydesc[awaydesc.str.contains('Time')]
    # timeout_type = df['TIMEOUT_TYPE'].dropna()
    df['remaining_in_quarter'] = pd.to_datetime(df['remaining_in_quarter'])
    last_play_before_2_mins = \
        df[df['remaining_in_quarter'] >= pd.datetime.today().replace(hour=2, minute=0, second=0)].groupby('game_id')[
            'remaining_in_quarter'].last().reset_index()
    last_play_before_2_mins['last_play_before_2_mins'] = True

    df = df.merge(last_play_before_2_mins, on=['game_id', 'remaining_in_quarter'], how='left')

    df['last_play_before_2_mins'].fillna(False, inplace=True)
    df['SCOREMARGIN'] = (df['home_score'] - df['away_score'].astype(np.int)).astype(np.int)

    final_home_margin = df.groupby('game_id')['SCOREMARGIN'].last().reset_index()
    final_margin_ix_game = final_home_margin.rename(columns={'SCOREMARGIN': 'final_score_margin'}).set_index('game_id')

    two_mins_home_margin = \
        df[df['last_play_before_2_mins']].drop_duplicates(subset='game_id', keep='last').set_index('game_id')[
            'SCOREMARGIN'].rename('two_mins_score_margin')

    two_min_outcomes = final_margin_ix_game.join(two_mins_home_margin, how='inner')

    two_min_outcomes['winning_team'] = np.where(
        two_min_outcomes['final_score_margin'] > 0,
        'home',
        'away'
    )

    two_min_outcomes['winning_team_two_min_margin'] = np.where(
        two_min_outcomes['final_score_margin'] > 0,
        two_min_outcomes['two_mins_score_margin'],
        two_min_outcomes['two_mins_score_margin'] * (-1)
    )
    two_min_outcomes['leading_team_won'] = np.where(
        two_min_outcomes['two_mins_score_margin'] > 0,
        np.where(two_min_outcomes['final_score_margin'] > 0, 1, 0),
        np.where(two_min_outcomes['final_score_margin'] < 0, 1, 0)
    )
    two_min_outcomes['abs_two_mins_score_margin'] = two_min_outcomes['two_mins_score_margin'].abs()
    return two_min_outcomes

def get_full_df(years_requested: list) -> pd.DataFrame:
    _filter_fxs = [
        lambda df: df[df['period'] == 4],
        lambda df: df[
            pd.to_datetime(df['remaining_in_quarter']) <= pd.datetime.today().replace(hour=3, minute=0, second=0)]
    ]
    df = load_clean_pbp(clean_dir=CLEAN_ROOT_DIR, filetype=CLEAN_OUTPUT_FILETYPE, years_requested=years_requested,
                        filter_fxs=_filter_fxs)

    # awaydesc = df['AWAYDESCRIPTION'].dropna()
    # awaytimeouts = awaydesc[awaydesc.str.contains('Time')]
    # timeout_type = df['TIMEOUT_TYPE'].dropna()
    df['remaining_in_quarter'] = pd.to_datetime(df['remaining_in_quarter'])
    last_play_before_2_mins = \
        df[df['remaining_in_quarter'] >= pd.datetime.today().replace(hour=2, minute=0, second=0)].groupby('game_id')[
            'remaining_in_quarter'].last().reset_index()
    last_play_before_2_mins['last_play_before_2_mins'] = True

    df = df.merge(last_play_before_2_mins, on=['game_id', 'remaining_in_quarter'], how='left')

    df['last_play_before_2_mins'].fillna(False, inplace=True)
    df['SCOREMARGIN'] = (df['home_score'] - df['away_score'].astype(np.int)).astype(np.int)

    return df

# _years_requested = [
#     [2008, 2009, 2010, 2011],
#     [2013, 2014, 2015],
#     [2017, 2018]
# ]
# %%
## Function to Merge dfs ##
def merging_df(agg_score_margin=9, _years_requested=_years_requested):

    agg_two_min_outcomes = pd.concat([get_two_minute_outcomes(years_requested=years) for years in _years_requested])

    agg_two_min_outcomes = agg_two_min_outcomes[abs(agg_two_min_outcomes['two_mins_score_margin'])<=agg_score_margin]
    agg_two_min_outcomes.reset_index(inplace=True)

    ## Running the Full Dataframe Data ##
    df = pd.concat([get_full_df(years_requested=years) for years in _years_requested])

    ## Isolating the Necessary Columns ##
    df = df[['game_id','play_id','HOMEDESCRIPTION','AWAYDESCRIPTION',
                        'home_score','away_score','remaining_in_quarter']]

    ## Splitting the Home/Away Score Margin ##
    df['HOMESCOREMARGIN'] = df['home_score']-df['away_score']
    df['AWAYSCOREMARGIN'] = -df['HOMESCOREMARGIN']

    ## Keeping all df Values with Valid Game ID ##
    df = df[df['game_id'].apply(lambda x: x in agg_two_min_outcomes['game_id'].unique())]

    ## Joining the Two Dataframes ##
    merged_df = df.merge(agg_two_min_outcomes, left_on='game_id', right_on='game_id', how='left')
    merged_df['AWAY_two_mins_score_margin'] = -merged_df['two_mins_score_margin']
    merged_df['HOME_two_mins_score_margin'] = merged_df['two_mins_score_margin']
    merged_df['time_remaining_seconds'] = merged_df['remaining_in_quarter'].apply(lambda x: x.hour*60+x.minute)

    games = []

    for i in merged_df['game_id'].unique()[0:2]:
        game = merged_df[merged_df['game_id'] == i]

        game['HOMETIMEOUTS'] = np.nan
        game['HOMETIMEOUTS'][game['play_id']==game['play_id'].min()] = 3

        count = 0
        timeouts_remaining = 3
        for i in range(0,len(game)):
            try:     
                if game.iloc[i, 2].lower().find('timeout') != -1:
                    timeouts_remaining -= 1
                    game.iloc[i, -1] = timeouts_remaining
            except:
                game.iloc[i, -1] = timeouts_remaining

        game['HOMETIMEOUTS'].fillna(method='ffill',inplace=True)

        game['AWAYTIMEOUTS'] = np.nan
        game['AWAYTIMEOUTS'][game['play_id']==game['play_id'].min()] = 3

        count = 0
        timeouts_remaining = 3
        for i in range(0,len(game)):
            try:     
                if game.iloc[i, 3].lower().find('timeout') != -1:
                    timeouts_remaining -= 1
                    game.iloc[i, -1] = timeouts_remaining
            except:
                game.iloc[i, -1] = timeouts_remaining

        game['AWAYTIMEOUTS'].fillna(method='ffill',inplace=True)

        games.append(game)

    games_ = pd.concat(games, ignore_index=True)

    games_ = games_.join(pd.get_dummies(games_['HOMETIMEOUTS'],prefix='HOME_TIMEOUT'))
    games_ = games_.join(pd.get_dummies(games_['AWAYTIMEOUTS'],prefix='AWAY_TIMEOUT'))
    games_.drop(['HOMETIMEOUTS','AWAYTIMEOUTS'], axis=1, inplace=True)
    
    merged_df = games_

    ## Splitting Home v. Away Won ##
    merged_df['home_won'] = 0
    merged_df['away_won'] = 0

    merged_df['home_won'][merged_df['winning_team']=='home'] = 1
    merged_df['away_won'][merged_df['winning_team']=='away'] = 1

    return merged_df

# %%
## Running the Agg Two Minute Outcome Data ##
_years_requested = [
    [2008, 2009]
    # [2008, 2009, 2010, 2011],
    # [2013, 2014, 2015],
    # [2017, 2018]
]

agg_two_min_outcomes = pd.concat([get_two_minute_outcomes(years_requested=years) for years in _years_requested])

win_rate_by_margin = agg_two_min_outcomes.groupby('abs_two_mins_score_margin')['leading_team_won'].mean()
print(win_rate_by_margin)

win_rate_by_margin.to_csv("win_rate_by_two_min_margin.csv")

agg_two_min_outcomes


# %%
## Splitting the Merged Dataframe to Home v. Away ##
merged_df_copy = merging_df().copy()
merged_df_copy.columns

merged_df_copy_home = merged_df_copy[[i for i in np.array(merged_df.columns) if 'home' in i.lower()][2:]]
merged_df_copy_away = merged_df_copy[[i for i in np.array(merged_df.columns) if 'away' in i.lower()][2:]]

# %%
## Scripts for Running Tests ##
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, roc_auc_score, classification_report

from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.decomposition import PCA

## Function for Question 2 - Logistic Regression ##
def log_reg(df, test_size=0.2, seed=2):
    
    # Setting Seed #
    np.random.seed(seed)

    # Extracting Features #
    X = df.iloc[:,:-1]
    y = df.iloc[:,-1]

    X_train, X_test, y_train, y_test = \
        train_test_split(X, y, test_size=test_size, random_state=42)
    
    # Running Model #
    lr = LogisticRegression()
    lr.fit(X_train, y_train)

    # Stats #
    score = lr.score(X_test, y_test)
    conf_matrix = confusion_matrix(y_test, lr.predict(X_test))
    cr = classification_report(y_test, lr.predict(X_test))

    print(f'The accuracy for the logistic regression model is {round(score,2)}')
    print()
    print(f'The logistic regression model confusion matrix:')
    print()
    print(conf_matrix)
    print(f'The logistic regression model classification report:')
    print()
    print(cr)

## Function for Question 3 - Comparing Multiple Classifiers ##
def comparing_models(df, test_size=0.2):

    # Extracting Features #
    X = df.iloc[:,:-1]
    y = df.iloc[:,-1]

    X_train, X_test, y_train, y_test = \
        train_test_split(X, y, test_size=test_size, random_state=42)

    # Setting up the Models #
    names = [
        "Logistic Regression",
        "K Nearest Neighbors", 
        "SVM",
        "KSVM",
        "Neural Network"
        ]

    classifiers = [
        LogisticRegression(),
        KNeighborsClassifier(),
        SVC(kernel='linear'),
        SVC(kernel='rbf'),
        MLPClassifier(hidden_layer_sizes=(20,10))
        ]

    # Creating a Dictionary for the Accuracies #
    accuracies = {}

    # Running the Process for the Three Classifiers #
    for name, clf in zip(names, classifiers):
        clf.fit(X_train, y_train)
        score = clf.score(X_test, y_test)
        conf_matrix = confusion_matrix(y_test, clf.predict(X_test))
        cr = classification_report(y_test, clf.predict(X_test))

        accuracies[name] = score

        ## Printing the Accuracies and the Confusion Matrix for Each of the Classifiers ##
        print(f'The accuracy for the {name} model is {round(score,2)}')
        print()
        print(f'The {name} model confusion matrix:')
        print()
        print(conf_matrix)
        print(f'The {name} model classification report:')
        print()
        print(cr)

    # Plotting the Accuracies #
    plt.figure(figsize=(10,5))
    plt.bar(accuracies.keys(),accuracies.values())

    for i, v in enumerate(accuracies.values()):
        plt.text(i-0.08, v-0.05, str(np.round(v,4)*100)+'%', color='w')

    plt.savefig('images/model_comparison.png')
# %%
log_reg(merged_df_copy_home)
comparing_models(merged_df_copy_home)

# %%
