# %% imports
import datetime as dt
import pandas as pd
import numpy as np

from nba_pbp_analysis.data.intermediate.base_intm_dataset import BaseIntermediateDataIO


# %% determine eventual winner (by gameid)

class WinnerData(BaseIntermediateDataIO):
    request_id = "winner_gameid"
    
    @classmethod
    def source(cls, by: str = "gameid", **kwargs) -> pd.DataFrame:
        if by == "gameid":
            return cls.source_by_gameid(**kwargs)
        else:
            raise NotImplementedError
    
    @classmethod
    def source_by_gameid(cls, years_requested: list, **kwargs) -> pd.DataFrame:
        df = cls.read_raw_data(years_requested=years_requested, **kwargs)
        final_margin_ix_game = cls._calc_final_margin(df=df)
        final_margin_ix_game['winning_team'] = np.where(final_margin_ix_game['final_score_margin'] > 0, "home", "away")
        return final_margin_ix_game
    
    @classmethod
    def _filter_fxs(cls, max_rem_in_quarter: dt.datetime = None, **kwargs) -> list:
        max_rem_in_quarter = dt.datetime(year=1900, month=1, day=1, hour=0, minute=0,
                                         second=1) if max_rem_in_quarter is None else max_rem_in_quarter
        return [
            lambda df: df[df['period'] == 4],
            # lambda df: df[pd.to_datetime(df['remaining_in_quarter']) <= max_remaining_in_quarter]
            lambda df: df[pd.to_datetime(df['remaining_in_quarter'], format="%M:%S") <= max_rem_in_quarter]
        ]
    
    @staticmethod
    def _calc_final_margin(df: pd.DataFrame) -> pd.DataFrame:
        df['SCOREMARGIN'] = (df['home_score'] - df['away_score'].astype(np.int)).astype(np.int)
        final_home_margin = df.groupby('game_id')['SCOREMARGIN'].last().reset_index()
        final_margin_ix_game = final_home_margin.rename(columns={'SCOREMARGIN': 'final_score_margin'}).set_index(
            'game_id')
        return final_margin_ix_game
