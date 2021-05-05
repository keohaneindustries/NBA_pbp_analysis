# %% imports
from abc import abstractmethod
import datetime as dt
import os
import pandas as pd
import sys
from collections import OrderedDict
from functools import reduce

from nba_pbp_analysis.data.cf import *


# %% functions


def _get_filepaths_to_read(clean_dir: str, filetype: str = ".csv", years_requested: list = None) -> list:
    assert os.path.isdir(clean_dir)
    
    # crawl clean_data_root_dir for filenames to read -> list of all filenames
    allfiles = [f for f in os.listdir(clean_dir) if os.path.isfile(os.path.join(clean_dir, f))]
    # filter list based on filetype suffix
    pbpfiles = [f for f in allfiles if f[-len(filetype):] == filetype]
    # filter list based on years
    good_files = [
        os.path.join(clean_dir, p) for p in pbpfiles if
        (int(p[-len(filetype) - 9:-len(filetype) - 5]) in years_requested) or (
                int(p[-len(filetype) - 4:-len(filetype)]) in years_requested)
    ]
    return good_files


def _read_file_to_df(filepath: str, filetype: str = ".csv") -> pd.DataFrame:
    """read file to dataframe"""
    print("reading file {} to df...".format(filepath))
    if filetype == ".csv":
        return pd.read_csv(filepath)
    elif filetype == ".pkl":
        return pd.read_pickle(filepath)
    else:
        raise NotImplementedError


def _apply_filters(df: pd.DataFrame, filter_fxs: list = None) -> pd.DataFrame:
    print("applying filters...")
    if filter_fxs is None:
        return df
    for filter_fx in filter_fxs:
        df = filter_fx(df).copy()
    return df


def _slice_to_cols(df: pd.DataFrame, master_cols_requested: list = None) -> pd.DataFrame:
    print("slicing to requested cols...")
    if master_cols_requested is None:
        return df
    return df.loc[:, master_cols_requested]


def _retype_data(df):
    """parse and re-type data as necessary (to match master spec's)"""
    print("re-typing data...")
    # raise NotImplementedError  # TODO
    df.loc[:, 'rem_in_quarter_dt'] = pd.to_datetime(df.loc[:, 'remaining_in_quarter'], format="%M:%S")
    return df


def _append_season_col(df: pd.DataFrame, filepath: str) -> pd.DataFrame:
    print("appending season col...")
    df["season"] = int(filepath[-8:-4])
    return df


def load_clean_pbp(clean_dir: str, filetype: str = ".csv", years_requested: list = None,
                   master_cols_requested: list = None, filter_fxs: list = None) -> pd.DataFrame:
    filepaths = _get_filepaths_to_read(clean_dir=clean_dir, filetype=filetype, years_requested=years_requested)
    return pd.concat([
        _append_season_col(
            df=_retype_data(
                df=_slice_to_cols(
                    df=_apply_filters(
                        df=_read_file_to_df(
                            filepath=filepath, filetype=filetype
                        ), filter_fxs=filter_fxs
                    ), master_cols_requested=master_cols_requested
                )
            ), filepath=filepath
        )
        for filepath in filepaths
    ]).reset_index(drop=True)


# %% examples

def main():
    print("loading example #1...")
    _years_requested = list(range(2010, 2015))
    _filter_fxs = [
        lambda df: df[df['player1'] == "LeBron James"],
        lambda df: df[(df['HOME_TEAM'] == "Heat") | (df['AWAY_TEAM'] == "Heat")],
    
    ]
    df = load_clean_pbp(clean_dir=CLEAN_ROOT_DIR, filetype=CLEAN_OUTPUT_FILETYPE, years_requested=_years_requested,
                        filter_fxs=_filter_fxs)
    print("completed loading example #1:\n{}\n\n".format(df))
    
    print("loading example #2...")
    _years_requested = list(range(2019, 2021))
    _filter_fxs = [
        lambda df: df[df['team1'] == "MIA"]
    ]
    df = load_clean_pbp(clean_dir=CLEAN_ROOT_DIR, filetype=CLEAN_OUTPUT_FILETYPE, years_requested=_years_requested,
                        filter_fxs=_filter_fxs)
    print("completed loading example #2:\n{}\n\n".format(df))
    
    game_ids = df['game_id'].unique()
    print("loading example #3...")
    _years_requested = list(range(2019, 2021))
    _filter_fxs = [
        lambda df: df[df['game_id'].isin(game_ids)]
    ]
    df = load_clean_pbp(clean_dir=CLEAN_ROOT_DIR, filetype=CLEAN_OUTPUT_FILETYPE, years_requested=_years_requested,
                        filter_fxs=_filter_fxs)
    print("completed loading example #3:\n{}\n\n".format(df))


if __name__ == '__main__':
    main()
