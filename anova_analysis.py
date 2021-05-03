# %% imports
import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt

from project_data import get_stored_anova_data


# %% clean anova

def _filter_anova_data(df_anova_data: pd.DataFrame) -> pd.DataFrame:
    df_anova_data = df_anova_data[df_anova_data['t_remaining_s'] > 0]  # drop observations where the game has ended
    # df_anova_data = df_anova_data[df_anova_data['t_remaining_s'] <= 60.]  # filter to last x seconds of data
    df_anova_data = df_anova_data[df_anova_data['t_remaining_s'] <= 60.]  # filter to last x seconds of data
    return df_anova_data


# %% anova fx

def run_anova(df_anova_data: pd.DataFrame, **kwargs):
    outcome_tree = df_anova_data.groupby(['lead_status', 'is_home', 'poss_follow_to'])['points_on_play'].describe()
    # outcome_tree = df_anova_data.groupby(['lead_status', 'poss_follow_to'])['points_on_play'].describe()
    print("outcome tree:\n{}".format(outcome_tree))
    raise NotImplementedError


# %% main
def main():
    df_anova_data = _filter_anova_data(df_anova_data=get_stored_anova_data())
    run_anova(df_anova_data=df_anova_data)
    return


if __name__ == '__main__':
    main()
