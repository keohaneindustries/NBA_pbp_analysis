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
RANDOM_SEED = 2
P_SIG_THRESHOLD = 0.05
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


# %% anova fx - 2-sided


def run_2s_anova(df_anova_data: pd.DataFrame, dep_var: str, cat_1: str, cat_2: str, downsample: bool = True,
                 p_sig: float = 0.05, **kwargs):
    if downsample is True:
        df_anova_data = downsample_anova_data(df_anova_data=df_anova_data, dep_var=dep_var, cat_vars=[cat_1, cat_2],
                                              **kwargs)
    formula = "{dep_var} ~ C({cat_1}) + C({cat_2}) + C({cat_2}):C({cat_1})".format(dep_var=dep_var, cat_1=cat_1,
                                                                                   cat_2=cat_2)
    print("fitting 2-sided ANOVA...")
    model = ols(formula, df_anova_data).fit()
    table = sm.stats.anova_lm(model, typ=2)
    table['signif'] = table["PR(>F)"] <= p_sig
    return table


def iteratively_run_2s_anova(df_anova_data: pd.DataFrame, dep_var: str = VAR_POINTS_ON_PLAY,
                             cat_1: str = VAR_POSS_FOLLOW_TO, cat_2: str = VAR_IS_HOME,
                             addtl_cat_to_slice: str = VAR_LEAD_STATUS, downsample: bool = True, **kwargs):
    table = run_2s_anova(df_anova_data=df_anova_data.copy(), dep_var=dep_var, cat_1=cat_1, cat_2=cat_2,
                         downsample=downsample, **kwargs)
    slice_name = "all_data" if addtl_cat_to_slice is None else "{}_all".format(addtl_cat_to_slice)
    print('2-sided ANOVA table for {}:\n{}\n'.format(slice_name, table))
    LocalIOUtils.save_csv(df=table, fileid="ANOVA_2s_model_{}".format(slice_name))
    
    if addtl_cat_to_slice is None:
        return
    
    cats_to_slice = df_anova_data[addtl_cat_to_slice].unique()
    for cat in cats_to_slice:
        _df = df_anova_data[df_anova_data[addtl_cat_to_slice] == cat].copy()
        table = run_2s_anova(df_anova_data=_df, dep_var=dep_var, cat_1=cat_1, cat_2=cat_2, downsample=downsample,
                             **kwargs)
        slice_name = "{}_{}".format(addtl_cat_to_slice, cat)
        print('2-sided ANOVA table for {}:\n{}\n'.format(slice_name, table))
        LocalIOUtils.save_csv(df=table, fileid="ANOVA_2s_model_{}".format(slice_name))
    return


# %% anova fx - 3-sided

def formula_anova_3s_DEFAULT(dep_var: str, cat_1: str, cat_2: str, cat_3: str) -> str:
    formula = """{dep_var} ~
            C({cat_1}, Sum) + C({cat_2}, Sum) + C({cat_3}, Sum) +
            C({cat_2}, Sum):C({cat_1}, Sum) + C({cat_3}, Sum):C({cat_1}, Sum) + C({cat_2}, Sum):C({cat_3}, Sum) +
            C({cat_2}, Sum):C({cat_3}, Sum):C({cat_1}, Sum)
            """.format(dep_var=dep_var, cat_1=cat_1, cat_2=cat_2, cat_3=cat_3)
    return formula


def formula_anova_3s_ITERATION_2(dep_var: str, cat_1: str, cat_2: str, cat_3: str) -> str:
    """ dropped the 3-sided combination term """
    formula = """{dep_var} ~
            C({cat_1}, Sum) + C({cat_2}, Sum) + C({cat_3}, Sum) +
            C({cat_2}, Sum):C({cat_1}, Sum) + C({cat_3}, Sum):C({cat_1}, Sum) + C({cat_2}, Sum):C({cat_3}, Sum)
            """.format(dep_var=dep_var, cat_1=cat_1, cat_2=cat_2, cat_3=cat_3)
    return formula


def formula_anova_3s_ITERATION_3(dep_var: str, cat_1: str, cat_2: str, cat_3: str) -> str:
    formula = """points_on_play ~
    C(is_home, Sum) + C(lead_status, Sum) + C(poss_follow_to, Sum) +
    C(is_home, Sum):C(poss_follow_to, Sum)
    """
    return formula


def formula_anova_3s_ITERATION_4(dep_var: str, cat_1: str, cat_2: str, cat_3: str) -> str:
    formula = '''points_on_play ~ C(lead_status, Sum) +
                C(is_home, Sum):C(poss_follow_to, Sum)'''
    return formula


def run_3s_anova(df_anova_data: pd.DataFrame, dep_var: str, cat_1: str, cat_2: str, cat_3: str, formula: str = None,
                 downsample: bool = True, model_id: str = None, p_sig: float = 0.05, **kwargs):
    if formula is None:
        formula_anova_3s_DEFAULT(dep_var, cat_1, cat_2, cat_3)
    if downsample is True:
        df_anova_data = downsample_anova_data(df_anova_data=df_anova_data, dep_var=dep_var,
                                              cat_vars=[cat_1, cat_2, cat_3], **kwargs)
    str_model_id = " ({})".format(model_id) if model_id is not None else ""
    print("fitting 3-sided ANOVA{}...".format(str_model_id))
    model = ols(formula, df_anova_data).fit()
    table = sm.stats.anova_lm(model, typ=3)
    table['signif'] = table["PR(>F)"] <= p_sig
    return table


def iteratively_run_3s_anova(df_anova_data: pd.DataFrame, dep_var: str = VAR_POINTS_ON_PLAY,
                             cat_1: str = VAR_POSS_FOLLOW_TO, cat_2: str = VAR_IS_HOME, cat_3: str = VAR_LEAD_STATUS,
                             downsample: bool = True, **kwargs):
    # iteration 1
    seed = 1
    table = run_3s_anova(df_anova_data=df_anova_data.copy(), dep_var=dep_var, cat_1=cat_1, cat_2=cat_2, cat_3=cat_3,
                         formula=formula_anova_3s_DEFAULT(dep_var, cat_1, cat_2, cat_3), downsample=downsample,
                         **kwargs)
    print('3-sided ANOVA table for all data:\n{}\n'.format(table))
    LocalIOUtils.save_csv(df=table, fileid="ANOVA_3s_model_all_factors")
    
    # iteration 2
    model_id = "iteration_2"
    seed = 2
    print("(dropped the 3-sided combination term)")
    table = run_3s_anova(df_anova_data=df_anova_data.copy(), dep_var=dep_var, cat_1=cat_1, cat_2=cat_2, cat_3=cat_3,
                         formula=formula_anova_3s_ITERATION_2(dep_var, cat_1, cat_2, cat_3), downsample=downsample,
                         model_id=model_id, seed=seed, **kwargs)
    print('3-sided ANOVA table ({}):\n{}\n'.format(model_id, table))
    LocalIOUtils.save_csv(df=table, fileid="ANOVA_3s_model_{}".format(model_id))
    
    # iteration 3
    model_id = "iteration_3"
    seed = 3
    table = run_3s_anova(df_anova_data=df_anova_data.copy(), dep_var=dep_var, cat_1=cat_1, cat_2=cat_2, cat_3=cat_3,
                         formula=formula_anova_3s_ITERATION_3(dep_var, cat_1, cat_2, cat_3), downsample=downsample,
                         model_id=model_id, seed=seed, **kwargs)
    print('3-sided ANOVA table ({}):\n{}\n'.format(model_id, table))
    LocalIOUtils.save_csv(df=table, fileid="ANOVA_3s_model_{}".format(model_id))
    
    # iteration 4
    model_id = "iteration_4"
    seed = 4
    table = run_3s_anova(df_anova_data=df_anova_data.copy(), dep_var=dep_var, cat_1=cat_1, cat_2=cat_2, cat_3=cat_3,
                         formula=formula_anova_3s_ITERATION_4(dep_var, cat_1, cat_2, cat_3), downsample=downsample,
                         model_id=model_id, seed=seed, **kwargs)
    print('3-sided ANOVA table ({}):\n{}\n'.format(model_id, table))
    LocalIOUtils.save_csv(df=table, fileid="ANOVA_3s_model_{}".format(model_id))


# %% run anova

def run_anova(df_anova_data: pd.DataFrame, seed: int = RANDOM_SEED, **kwargs):
    iteratively_run_2s_anova(df_anova_data=df_anova_data, dep_var=VAR_POINTS_ON_PLAY, cat_1=VAR_POSS_FOLLOW_TO,
                             cat_2=VAR_IS_HOME, addtl_cat_to_slice=VAR_LEAD_STATUS, seed=seed, **kwargs)
    
    iteratively_run_2s_anova(df_anova_data=df_anova_data, dep_var=VAR_POINTS_ON_PLAY, cat_1=VAR_POSS_FOLLOW_TO,
                             cat_2=VAR_LEAD_STATUS, addtl_cat_to_slice=VAR_IS_HOME, seed=seed, **kwargs)
    
    iteratively_run_3s_anova(df_anova_data=df_anova_data.copy(), dep_var=VAR_POINTS_ON_PLAY, cat_1=VAR_POSS_FOLLOW_TO,
                             cat_2=VAR_LEAD_STATUS, cat_3=VAR_IS_HOME, **kwargs)
    
    return


# %% main
def main():
    df_anova_data = _filter_anova_data(df_anova_data=get_stored_anova_data())
    _ = get_outcome_tree(df_anova_data=df_anova_data, dep_var=VAR_POINTS_ON_PLAY, cat_vars=GLOBAL_CAT_VARS,
                         sample_id='population')
    
    run_anova(df_anova_data=df_anova_data, seed=RANDOM_SEED, p_sig=P_SIG_THRESHOLD)
    return


if __name__ == '__main__':
    main()
