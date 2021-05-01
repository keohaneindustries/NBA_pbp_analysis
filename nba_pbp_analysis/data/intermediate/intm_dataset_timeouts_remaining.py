# %% imports
import datetime as dt
import pandas as pd
import numpy as np

from nba_pbp_analysis.data.intermediate.base_intm_dataset import BaseIntermediateDataIO


# %% determine timeouts remaining by playid

class TimeoutsRemainingData(BaseIntermediateDataIO):
    request_id = "timeoutsremaining_playid"
    
    @classmethod
    def _filter_fxs(cls, **kwargs) -> list:
        return [
            lambda df: df[(df['period'] == 3) | (df['period'] == 4)]
            # lambda df: df[pd.to_datetime(df['remaining_in_quarter']) <= max_remaining_in_quarter]
            # lambda df: df[pd.to_datetime(df['remaining_in_quarter'], format="%M:%S") <= max_rem_in_quarter]
        ]
    
    @classmethod
    def source(cls, years_requested: list, **kwargs) -> pd.DataFrame:
        ## input
        # get all plays 4th quarter
        df = cls.read_raw_data(years_requested=years_requested, **kwargs)
        # slice to minimum variables required
        df = cls._slice_to_relevant_vars_for_timeout_remaining_backfill(df=df)
        
        ## calcs
        # calc timeouts remaining for each play
        df = cls._calc_timeouts_remaining(df=df)
        
        # slice to minimum timespan required in final output
        max_rem_in_quarter = dt.datetime(year=1900, month=1, day=1, hour=0, minute=3, second=0)
        df = cls._slice_to_relevant_timespan(df=df, max_rem_in_quarter=max_rem_in_quarter, period=4)
        
        df = cls._enforce_4th_period_max_to(df=df)
        
        ## output
        # slice to minimum variables required in output
        df = cls._slice_to_output_vars(df=df)
        # set index for uniform .csv file I/O
        df = cls._reindex_game_play_id(df=df)
        
        return df
    
    @staticmethod
    def _slice_to_relevant_vars_for_timeout_remaining_backfill(df: pd.DataFrame) -> pd.DataFrame:
        relevant_vars = ['game_id', 'play_id', 'HOMEDESCRIPTION', 'AWAYDESCRIPTION', 'period', 'rem_in_quarter_dt',
                         'season']
        return df.loc[:, relevant_vars]
    
    @classmethod
    def _calc_timeouts_remaining(cls, df: pd.DataFrame) -> pd.DataFrame:
        df = cls._calc_timeouts_allowed(df=df)
        home_timeouts = cls._calc_timeouts_used_by_side(df=df, side="home")
        away_timeouts = cls._calc_timeouts_used_by_side(df=df, side="away")
        
        timeouts = df[['game_id', 'play_id', 'period', 'rem_in_quarter_dt']].join(home_timeouts).join(away_timeouts)
        timeouts = timeouts.set_index('game_id', append=True).groupby(level=1).ffill().reset_index(level=1)
        return timeouts
    
    @classmethod
    def _calc_timeouts_allowed(cls, df: pd.DataFrame):
        df['to_n_allowed'] = np.where(df['season'] >= 2017, 7, 8)
        return df
    
    @classmethod
    def _calc_timeouts_used_by_side(cls, df: pd.DataFrame, side: str = "home") -> pd.DataFrame:
        if side == "home":
            col_name = "HOMEDESCRIPTION"
        elif side == "away":
            col_name = "AWAYDESCRIPTION"
        else:
            raise ValueError
        str_timeouts = cls._find_timeouts(df=df, col_name=col_name)
        n_timeouts = str_timeouts[col_name].str[10:].str.extract('(\d)(?:\D+)(\d)').fillna(-1).astype(np.int)
        n_timeouts.columns = ['to_n_full', 'to_n_short']
        n_timeouts['to_n_total'] = n_timeouts['to_n_full'] + n_timeouts['to_n_short']
        
        n_timeouts = n_timeouts.join(df[['to_n_allowed']])
        n_timeouts['to_rem'] = n_timeouts['to_n_allowed'] - n_timeouts['to_n_total']
        n_timeouts[n_timeouts['to_n_total'] < 0] = np.nan
        n_timeouts.drop('to_n_allowed', axis=1, inplace=True)
        n_timeouts.columns = [side + "_" + col for col in n_timeouts.columns]
        return n_timeouts.copy()
    
    @classmethod
    def _find_timeouts(cls, df: pd.DataFrame, col_name: str) -> pd.DataFrame:
        return df[df[col_name].str.contains("Timeout").fillna(False)]
    
    @staticmethod
    def _slice_to_relevant_timespan(df: pd.DataFrame, max_rem_in_quarter: dt.datetime, period: int = 4) -> pd.DataFrame:
        return df[(df['rem_in_quarter_dt'] <= max_rem_in_quarter) & (df['period'] == period)].dropna()
    
    @classmethod
    def _enforce_4th_period_max_to(cls, df: pd.DataFrame) -> pd.DataFrame:
        t_rem_pre_2017 = dt.datetime(year=1900, month=1, day=1, hour=0, minute=3, second=0)
        t_rem_post_2017 = dt.datetime(year=1900, month=1, day=1, hour=0, minute=2, second=0)
        for col in ['home_to_rem', 'away_to_rem']:
            df[col] = np.where(
                df['season'] < 2017,
                np.where(
                    (df['rem_in_quarter_dt'] <= t_rem_pre_2017) & (df['period'] == 4),
                    np.column_stack([df[col].values, np.full_like(df[col].values, 3)]).min(axis=1),
                    df[col]
                ),
                np.where(
                    (df['rem_in_quarter_dt'] <= t_rem_post_2017) & (df['period'] == 4),
                    np.column_stack([df[col].values, np.full_like(df[col].values, 2)]).min(axis=1),
                    df[col]
                )
            )
        return df
    
    @staticmethod
    def _slice_to_output_vars(df: pd.DataFrame) -> pd.DataFrame:
        op_col_suffixes = ['_to_n_full', '_to_n_short', '_to_n_total', '_to_rem']
        relevant_vars = ['game_id', 'play_id'] + ["home" + suf for suf in op_col_suffixes] + ["away" + suf for suf in
                                                                                              op_col_suffixes]
        return df.loc[:, relevant_vars]
    
    @staticmethod
    def _reindex_game_play_id(df: pd.DataFrame) -> pd.DataFrame:
        df['game_play_id'] = df[['game_id', 'play_id']].astype(np.int).apply(
            lambda row: "{}_{}".format(row['game_id'], row['play_id']), axis=1)
        df.set_index('game_play_id', inplace=True)
        return df
