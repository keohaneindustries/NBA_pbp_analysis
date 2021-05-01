# %% imports
import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt

from project_data import get_stored_anova_data


# %% anova fx

def run_anova(df_anova_data: pd.DataFrame, **kwargs):
    outcome_tree = df_anova_data.groupby(['is_home', 'poss_follow_to'])['points_on_play'].describe()
    print("outcome tree:\n{}".format(outcome_tree))
    raise NotImplementedError


# %% main
def main():
    df_anova_data = get_stored_anova_data()
    run_anova(df_anova_data=df_anova_data)
    return


if __name__ == '__main__':
    main()
