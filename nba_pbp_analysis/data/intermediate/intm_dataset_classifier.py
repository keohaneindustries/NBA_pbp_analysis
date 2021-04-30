# %% imports
import datetime as dt
import pandas as pd
import numpy as np

from nba_pbp_analysis.data.intermediate.base_intm_dataset import BaseIntermediateDataIO


# %% determine eventual winner (by gameid)

class ClassifierData(BaseIntermediateDataIO):
    request_id = "classifier_playid"

    @classmethod
    def _filter_fxs(cls, max_rem_in_quarter: dt.datetime, **kwargs) -> list:
        return [
            lambda df: df[df['period'] == 4],
            # lambda df: df[pd.to_datetime(df['remaining_in_quarter']) <= max_remaining_in_quarter]
            lambda df: df[pd.to_datetime(df['remaining_in_quarter'], format="%M:%S") <= max_rem_in_quarter]
        ]

    @classmethod
    def source(cls, years_requested: list, df_winners=None, df_close_games=None, df_timeouts_remaining=None,
               **kwargs) -> pd.DataFrame:
        ## input
        # get all plays last 3 mins
        max_rem_in_quarter = dt.datetime(year=1900, month=1, day=1, hour=0, minute=3, second=0)
        df = cls.read_raw_data(years_requested=years_requested, max_rem_in_quarter=max_rem_in_quarter, **kwargs)
        # slice to minimum variables required
        df = cls._slice_to_relevant_vars_for_classifier_calcs(df=df)

        ## calcs
        # calc for each play: point margin, time remaining (s)
        df = cls._calc_derived_variables(df=df)
        # drop duplicates
        df = df.drop_duplicates()

        ## merging datasets
        # TODO left join df_closegame
        # TODO filter to close game == True
        df = cls._filter_to_close_games(df=df)
        # TODO left join df_winners
        # TODO left join df_timeouts_remaining

        ## transform home/away
        df = cls._transform_home_vs_away(df=df)

        ## output
        # TODO rename variables
        df = cls._rename_vars(df=df)
        # TODO slice to minimum variables required in output
        df = cls._slice_to_output_vars(df=df)
        return df

    @staticmethod
    def _slice_to_relevant_vars_for_classifier_calcs(df: pd.DataFrame) -> pd.DataFrame:
        relevant_vars = ['game_id', 'play_id', 'home_score', 'away_score', 'rem_in_quarter_dt']
        return df.loc[:, relevant_vars]

    # region derived variables
    @classmethod
    def _calc_derived_variables(cls, df: pd.DataFrame) -> pd.DataFrame:
        df = cls._calc_margin(df=df)
        df = cls._calc_t_remaining_s(df=df)
        return df

    @staticmethod
    def _calc_margin(df: pd.DataFrame) -> pd.DataFrame:
        df['SCOREMARGIN'] = (df['home_score'] - df['away_score'].astype(np.int)).astype(np.int)
        return df

    @staticmethod
    def _calc_t_remaining_s(df: pd.DataFrame) -> pd.DataFrame:
        df['t_remaining_s'] = (df['rem_in_quarter_dt'] - dt.datetime(year=1900, month=1, day=1, hour=0, minute=0,
                                                                     second=0)).dt.total_seconds().astype(int)
        return df

    # endregion

    @staticmethod
    def _filter_to_close_games(df: pd.DataFrame) -> pd.DataFrame:
        # return df
        raise NotImplementedError

    @classmethod
    def _transform_home_vs_away(cls, df: pd.DataFrame) -> pd.DataFrame:
        # TODO split into home/away
        df_home, df_away = cls._split_home_away(df=df)
        # TODO apply any necessary transformations to align variables
        df_home = cls._align_home_vars(df=df_home)
        df_away = cls._align_away_vars(df=df_away)
        # stack home/away
        df_restacked = pd.concat([df_home, df_away])
        # TODO calc for each play "did eventually win game" (bool)
        df_restacked = cls._calc_did_win_game(df=df_restacked)
        return df_restacked

    @staticmethod
    def _split_home_away(df: pd.DataFrame) -> tuple:
        # return df_home, df_away
        raise NotImplementedError

    @staticmethod
    def _align_home_vars(df: pd.DataFrame) -> pd.DataFrame:
        # return df
        raise NotImplementedError

    @staticmethod
    def _align_away_vars(df: pd.DataFrame) -> pd.DataFrame:
        # return df
        raise NotImplementedError

    @staticmethod
    def _calc_did_win_game(df: pd.DataFrame) -> pd.DataFrame:
        # return df
        raise NotImplementedError

    @staticmethod
    def _rename_vars(df: pd.DataFrame) -> pd.DataFrame:
        var_map = {}
        # return df.rename(columns=var_map)
        raise NotImplementedError

    @staticmethod
    def _slice_to_output_vars(df: pd.DataFrame) -> pd.DataFrame:
        relevant_vars = ['game_id', 'play_id', 'timeouts_rem_home', 'timeouts_rem_away']
        # return df.loc[:, relevant_vars]
        raise NotImplementedError
