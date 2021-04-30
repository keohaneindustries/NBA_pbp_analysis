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
            lambda df: df[df['period'] == 4]
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
        # TODO troubleshoot df_timeouts_remaining
        df = cls._calc_timeouts_remaining(df=df)
        
        # df = cls._calc_home_timeouts_remaining(df=df)
        # df = cls._calc_away_timeouts_remaining(df=df)
        
        ## output
        # slice to minimum timespan required in final output
        max_rem_in_quarter = dt.datetime(year=1900, month=1, day=1, hour=0, minute=3, second=0)
        df = cls._slice_to_relevant_timespan(df=df, max_rem_in_quarter=max_rem_in_quarter)
        # rename variables
        df = cls._rename_vars(df=df)
        # slice to minimum variables required in output
        df = cls._slice_to_output_vars(df=df)
        # set index for uniform .csv file I/O
        df = cls._reindex_game_play_id(df=df)
        
        return df
    
    @staticmethod
    def _slice_to_relevant_vars_for_timeout_remaining_backfill(df: pd.DataFrame) -> pd.DataFrame:
        relevant_vars = ['game_id', 'play_id', 'HOMEDESCRIPTION', 'AWAYDESCRIPTION', 'rem_in_quarter_dt']
        return df.loc[:, relevant_vars]
    
    @classmethod
    def _calc_timeouts_remaining(cls, df: pd.DataFrame) -> pd.DataFrame:
        l_games = []
        
        for i in df['game_id'].unique()[0:2]:
            game = df[df['game_id'] == i].copy()
            
            game['HOMETIMEOUTS'] = np.nan
            game['HOMETIMEOUTS'][game['play_id'] == game['play_id'].min()] = 3
            
            count = 0
            timeouts_remaining = 3
            for i in range(0, len(game)):
                try:
                    if game.iloc[i, 2].lower().find('timeout') != -1:
                        timeouts_remaining -= 1
                        game.iloc[i, -1] = timeouts_remaining
                except:
                    game.iloc[i, -1] = timeouts_remaining
            
            game['HOMETIMEOUTS'].fillna(method='ffill', inplace=True)
            
            game['AWAYTIMEOUTS'] = np.nan
            game['AWAYTIMEOUTS'][game['play_id'] == game['play_id'].min()] = 3
            
            count = 0
            timeouts_remaining = 3
            for i in range(0, len(game)):
                try:
                    if game.iloc[i, 3].lower().find('timeout') != -1:
                        timeouts_remaining -= 1
                        game.iloc[i, -1] = timeouts_remaining
                except:
                    game.iloc[i, -1] = timeouts_remaining
            
            game['AWAYTIMEOUTS'].fillna(method='ffill', inplace=True)
            
            l_games.append(game.copy())
        
        df_games = pd.concat(l_games, ignore_index=True)
        
        # df_games = df_games.join(pd.get_dummies(df_games['HOMETIMEOUTS'], prefix='HOME_TIMEOUT'))
        # df_games = df_games.join(pd.get_dummies(df_games['AWAYTIMEOUTS'], prefix='AWAY_TIMEOUT'))
        # df_games.drop(['HOMETIMEOUTS', 'AWAYTIMEOUTS'], axis=1, inplace=True)
        
        return df_games
    
    @classmethod
    def _calc_home_timeouts_remaining(cls, df: pd.DataFrame) -> pd.DataFrame:
        # TODO _calc_home_timeouts_remaining
        return df
    
    @classmethod
    def _calc_away_timeouts_remaining(cls, df: pd.DataFrame) -> pd.DataFrame:
        # TODO _calc_away_timeouts_remaining
        return df
    
    @staticmethod
    def _slice_to_relevant_timespan(df: pd.DataFrame, max_rem_in_quarter: dt.datetime) -> pd.DataFrame:
        return df[df['rem_in_quarter_dt'] <= max_rem_in_quarter]
    
    @staticmethod
    def _rename_vars(df: pd.DataFrame) -> pd.DataFrame:
        var_map = {'HOMETIMEOUTS': 'timeouts_rem_home', 'AWAYTIMEOUTS': 'timeouts_rem_away'}
        return df.rename(columns=var_map)
    
    @staticmethod
    def _slice_to_output_vars(df: pd.DataFrame) -> pd.DataFrame:
        relevant_vars = ['game_id', 'play_id', 'timeouts_rem_home', 'timeouts_rem_away']
        return df.loc[:, relevant_vars]
    
    @staticmethod
    def _reindex_game_play_id(df: pd.DataFrame) -> pd.DataFrame:
        df['game_play_id'] = df[['game_id', 'play_id']].astype(np.int).apply(
            lambda row: "{}_{}".format(row['game_id'], row['play_id']), axis=1)
        df.set_index('game_play_id', inplace=True)
        return df
