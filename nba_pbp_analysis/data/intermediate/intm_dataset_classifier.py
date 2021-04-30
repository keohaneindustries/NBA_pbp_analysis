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
        # calc for each play: point margin, time remaining (s), etc.
        df = cls._calc_derived_variables(df=df)
        # drop duplicates
        df = df.drop_duplicates()
        
        ## merging datasets
        df = df.merge(df_close_games, how='left', on='game_id')
        df = cls._filter_to_close_games(df=df)
        df = df.merge(df_winners, how='left', on='game_id')
        df = df.merge(df_timeouts_remaining, how='left', on=['game_id', 'play_id'])
        
        ## removing extraneous/duplicate data
        df = cls._slice_to_relevant_vars_for_home_away_split(df=df)
        df = cls._drop_materially_redundant_plays(df=df)
        df.dropna(inplace=True)  # TODO remove dropna after troubleshooting df_timeouts_remaining
        assert df.isna().sum().sum() == 0
        
        ## transform home/away
        df = cls._transform_home_vs_away(df=df)
        
        ## output
        df = cls._slice_to_output_vars(df=df)
        df.reset_index(drop=True, inplace=True)
        df.index.name = "obs_id"
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
        df = cls._calc_game_points_per_min(df=df)
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
    
    @staticmethod
    def _calc_game_points_per_min(df: pd.DataFrame) -> pd.DataFrame:
        """ avg points scored per minute in the game to that point """
        df['game_points_per_min'] = (df['home_score'] + df['away_score']) * 60. / (
                (df['t_remaining_s'] - 48 * 60) * (-1))
        return df
    
    # endregion
    
    @staticmethod
    def _filter_to_close_games(df: pd.DataFrame) -> pd.DataFrame:
        return df[df['close_game']]
    
    @staticmethod
    def _slice_to_relevant_vars_for_home_away_split(df: pd.DataFrame) -> pd.DataFrame:
        relevant_vars = ['game_id', 'play_id', 't_remaining_s', 'game_points_per_min', 'winning_team',
                         'home_score', 'away_score', 'timeouts_rem_home', 'timeouts_rem_away']
        return df.loc[:, relevant_vars]
    
    @staticmethod
    def _drop_materially_redundant_plays(df: pd.DataFrame) -> pd.DataFrame:
        return df.drop_duplicates(subset=[col for col in df.columns if col != "play_id"])
    
    @classmethod
    def _transform_home_vs_away(cls, df: pd.DataFrame) -> pd.DataFrame:
        df_home, df_away = cls._split_home_away(df=df)
        df_home = cls._align_home_vars(df=df_home)
        df_away = cls._align_away_vars(df=df_away)
        # stack home/away
        df_restacked = pd.concat([df_home, df_away]).reset_index(drop=True)
        # post-stacking calcs
        df_restacked = cls._calc_rel_score_margin(df=df_restacked)
        df_restacked = cls._calc_did_win_game(df=df_restacked)
        return df_restacked
    
    @staticmethod
    def _split_home_away(df: pd.DataFrame) -> tuple:
        # return df_home, df_away
        df_home = df.copy()
        df_home['team_side'] = "home"
        df_away = df.copy()
        df_away['team_side'] = "away"
        return df_home, df_away
    
    @staticmethod
    def _align_home_vars(df: pd.DataFrame) -> pd.DataFrame:
        return df.rename(columns={'timeouts_rem_home': 'team_rem_timeouts', 'timeouts_rem_away': 'opp_rem_timeouts',
                                  'home_score': 'team_score', 'away_score': 'opp_score'})
    
    @staticmethod
    def _align_away_vars(df: pd.DataFrame) -> pd.DataFrame:
        return df.rename(columns={'timeouts_rem_away': 'team_rem_timeouts', 'timeouts_rem_home': 'opp_rem_timeouts',
                                  'away_score': 'team_score', 'home_score': 'opp_score'})
    
    @staticmethod
    def _calc_rel_score_margin(df: pd.DataFrame) -> pd.DataFrame:
        df['rel_score_margin'] = df['team_score'] - df['opp_score']
        return df
    
    @staticmethod
    def _calc_did_win_game(df: pd.DataFrame) -> pd.DataFrame:
        df['did_win_game'] = df['team_side'] == df['winning_team']
        return df
    
    @staticmethod
    def _slice_to_output_vars(df: pd.DataFrame) -> pd.DataFrame:
        relevant_vars = ['did_win_game', 't_remaining_s', 'game_points_per_min', 'rel_score_margin',
                         'team_rem_timeouts',
                         'opp_rem_timeouts']
        return df.loc[:, relevant_vars]
