# %% imports
import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.formula.api import ols

from nba_pbp_analysis.data.local_io_utils import LocalIOUtils
from project_data import get_stored_anova_data

# %% globals
VAR_POINTS_ON_PLAY = 'points_on_play'
VAR_LEAD_STATUS = 'lead_status'
VAR_POSS_FOLLOW_TO = 'poss_follow_to'
VAR_IS_HOME = 'is_home'
GLOBAL_CAT_VARS = [VAR_LEAD_STATUS, VAR_IS_HOME, VAR_POSS_FOLLOW_TO]


# %% clean anova

def _filter_anova_data(df_anova_data: pd.DataFrame) -> pd.DataFrame:
    df_anova_data = df_anova_data[df_anova_data['t_remaining_s'] > 0]  # drop observations where the game has ended
    # df_anova_data = df_anova_data[df_anova_data['t_remaining_s'] <= 60.]  # filter to last x seconds of data
    df_anova_data = df_anova_data[df_anova_data['t_remaining_s'] <= 30.]  # filter to last x seconds of data
    return df_anova_data


# %% sampling fx

def get_outcome_tree(df_anova_data: pd.DataFrame, dep_var: str, cat_vars: list = None,
                     sample_id: str = None, **kwargs) -> pd.DataFrame:
    cat_vars = GLOBAL_CAT_VARS if cat_vars is None else cat_vars
    df_outcome_tree = df_anova_data.groupby(cat_vars)[dep_var].describe()
    if sample_id is not None:
        print("outcome tree ({}):\n{}".format(sample_id, df_outcome_tree))
        LocalIOUtils.save_csv(df=df_outcome_tree, fileid="ANOVA_outcome_tree_{}".format(sample_id))
    return df_outcome_tree


def _downsample_slice(df_slice: pd.DataFrame, size: int, seed: int = None, **kwargs):
    np.random.seed(seed=seed)
    return df_slice.sample(n=size, replace=False)


def downsample_anova_data(df_anova_data: pd.DataFrame, dep_var: str, cat_vars: list = None, **kwargs) -> pd.DataFrame:
    df_outcome_tree = get_outcome_tree(df_anova_data=df_anova_data, dep_var=dep_var, cat_vars=cat_vars, **kwargs)
    min_bucket_size = int(df_outcome_tree['count'].min())
    ix_buckets = pd.concat(
        [pd.DataFrame(df_outcome_tree.index.get_level_values(nam)) for nam in df_outcome_tree.index.names], axis=1)
    print("downsampling .... (min bucket size = {})".format(min_bucket_size))
    df_downsampled = pd.concat([
        _downsample_slice(df_slice=df_anova_data[(df_anova_data[cat_vars] == bucket.values).all(axis=1)],
                          size=min_bucket_size, **kwargs)
        for _, bucket in ix_buckets.iterrows()
    ], axis=0)
    return df_downsampled.copy()


# %% anova fx


def run_2s_anova(df_anova_data: pd.DataFrame, dep_var: str, cat_1: str, cat_2: str, downsample: bool = True, **kwargs):
    if downsample is True:
        df_anova_data = downsample_anova_data(df_anova_data=df_anova_data, dep_var=dep_var, cat_vars=[cat_1, cat_2],
                                              **kwargs)
    formula = "{dep_var} ~ C({cat_1}) + C({cat_2}) + C({cat_2}):C({cat_1})".format(dep_var=dep_var, cat_1=cat_1,
                                                                                   cat_2=cat_2)
    print("fitting ANOVA...")
    model = ols(formula, df_anova_data).fit()
    table = sm.stats.anova_lm(model, typ=2)
    return table


def iteratively_run_2s_anova(df_anova_data: pd.DataFrame, dep_var: str = VAR_POINTS_ON_PLAY,
                             cat_1: str = VAR_POSS_FOLLOW_TO, cat_2: str = VAR_IS_HOME,
                             addtl_cat_to_slice: str = VAR_LEAD_STATUS, downsample: bool = True):
    table = run_2s_anova(df_anova_data=df_anova_data.copy(), dep_var=dep_var, cat_1=cat_1, cat_2=cat_2,
                         downsample=downsample)
    slice_name = "all data" if addtl_cat_to_slice is None else addtl_cat_to_slice
    print('2-sided ANOVA table for {}:\n{}\n'.format(slice_name, table))
    
    if addtl_cat_to_slice is None:
        return
    
    cats_to_slice = df_anova_data[addtl_cat_to_slice].unique()
    for cat in cats_to_slice:
        _df = df_anova_data[df_anova_data[addtl_cat_to_slice] == cat].copy()
        table = run_2s_anova(df_anova_data=_df, dep_var=dep_var, cat_1=cat_1, cat_2=cat_2, downsample=downsample)
        slice_name = "{}='{}'".format(addtl_cat_to_slice, cat)
        print('2-sided ANOVA table for {}:\n{}\n'.format(slice_name, table))
    return


def run_anova(df_anova_data: pd.DataFrame, **kwargs):
    iteratively_run_2s_anova(df_anova_data=df_anova_data, dep_var=VAR_POINTS_ON_PLAY, cat_1=VAR_POSS_FOLLOW_TO,
                             cat_2=VAR_IS_HOME, addtl_cat_to_slice=VAR_LEAD_STATUS, **kwargs)
    
    iteratively_run_2s_anova(df_anova_data=df_anova_data, dep_var=VAR_POINTS_ON_PLAY, cat_1=VAR_POSS_FOLLOW_TO,
                             cat_2=VAR_LEAD_STATUS, addtl_cat_to_slice=VAR_IS_HOME, **kwargs)
    return


# %% main
def main():
    df_anova_data = _filter_anova_data(df_anova_data=get_stored_anova_data())
    _ = get_outcome_tree(df_anova_data=df_anova_data, dep_var=VAR_POINTS_ON_PLAY, cat_vars=GLOBAL_CAT_VARS,
                         sample_id='population')
    
    run_anova(df_anova_data=df_anova_data)
    return


if __name__ == '__main__':
    main()
