# %% imports
import datetime as dt
import pandas as pd
import numpy as np

from nba_pbp_analysis.data.intermediate.base_intm_dataset import BaseIntermediateDataIO


# %% determine "close games" (by playid and/or by gameid)


class CloseGamesByGameIDData(BaseIntermediateDataIO):
    request_id = "closegames_gameid"
    
    @classmethod
    def _filter_fxs(cls, max_rem_in_quarter: dt.datetime = None, **kwargs) -> list:
        max_rem_in_quarter = dt.datetime(year=1900, month=1, day=1, hour=0, minute=3,
                                         second=0) if max_rem_in_quarter is None else max_rem_in_quarter
        return [
            lambda df: df[df['period'] == 4],
            # lambda df: df[pd.to_datetime(df['remaining_in_quarter']) <= max_remaining_in_quarter]
            lambda df: df[pd.to_datetime(df['remaining_in_quarter'], format="%M:%S") <= max_rem_in_quarter]
        ]
    
    @classmethod
    def source(cls, years_requested: list, **kwargs) -> pd.DataFrame:
        df = cls.read_raw_data(years_requested=years_requested, **kwargs)
        df = cls._calc_margin(df=df)
        
        max_abs_margin = 9
        x_mins = 2
        
        col_last_x_mins = 'last_play_before_{}_mins'.format(x_mins)
        df = cls._flag_last_play_before_x_mins(df=df, x_mins=x_mins, col_name=col_last_x_mins)
        df_x_mins_margin = cls._calc_x_mins_home_margin(df=df, col_name=col_last_x_mins)
        
        col_score_margin_x_mins = "score_margin_{}_mins".format(x_mins)
        df_x_mins_margin = cls._rename_margin_cols(df_x_mins_margin=df_x_mins_margin,
                                                   col_score_margin_x_mins=col_score_margin_x_mins)
        
        # flag close games
        df_x_mins_margin['close_game'] = df_x_mins_margin['abs_{}'.format(col_score_margin_x_mins)] <= max_abs_margin
        assert df_x_mins_margin.index.name == "game_id"
        return df_x_mins_margin
    
    @staticmethod
    def _calc_margin(df: pd.DataFrame) -> pd.DataFrame:
        df['SCOREMARGIN'] = (df['home_score'] - df['away_score'].astype(np.int)).astype(np.int)
        return df
    
    @staticmethod
    def _flag_last_play_before_x_mins(df: pd.DataFrame, x_mins: int = 2,
                                      col_name: str = 'last_play_before_2_mins') -> pd.DataFrame:
        last_play_before_x_mins = \
            df[df['rem_in_quarter_dt'] >= dt.datetime(year=1900, month=1, day=1, hour=0, minute=x_mins)].groupby(
                'game_id')[
                'rem_in_quarter_dt'].last().reset_index()
        last_play_before_x_mins[col_name] = True
        
        df = df.merge(last_play_before_x_mins, on=['game_id', 'rem_in_quarter_dt'], how='left')
        
        df[col_name].fillna(False, inplace=True)
        return df
    
    @staticmethod
    def _calc_x_mins_home_margin(df: pd.DataFrame, col_name: str = 'last_play_before_2_mins') -> pd.DataFrame:
        return df[df[col_name]].drop_duplicates(subset='game_id', keep='last').set_index('game_id')[['SCOREMARGIN']]
    
    @staticmethod
    def _rename_margin_cols(df_x_mins_margin: pd.DataFrame, col_score_margin_x_mins: str) -> pd.DataFrame:
        df_x_mins_margin = df_x_mins_margin.rename(columns={"SCOREMARGIN": col_score_margin_x_mins})
        df_x_mins_margin['abs_{}'.format(col_score_margin_x_mins)] = df_x_mins_margin[col_score_margin_x_mins].abs()
        df_x_mins_margin.rename(columns={col_score_margin_x_mins: col_score_margin_x_mins + "_home"}, inplace=True)
        return df_x_mins_margin


class CloseGamesByPlayIDData(CloseGamesByGameIDData):
    
    @classmethod
    def source(cls, **kwargs) -> pd.DataFrame:
        raise NotImplementedError
