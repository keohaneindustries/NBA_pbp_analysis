# %% imports
import datetime as dt
import os
import pandas as pd
import sys
import numpy as np

from nba_pbp_analysis.data.cf import *
from nba_pbp_analysis.data.load import load_clean_pbp


# %%
def get_two_minute_outcomes(years_requested: list) -> pd.DataFrame:
    _filter_fxs = [
        lambda df: df[df['period'] == 4],
        lambda df: df[
            pd.to_datetime(df['remaining_in_quarter']) <= pd.datetime.today().replace(hour=3, minute=0, second=0)]
    ]
    df = load_clean_pbp(clean_dir=CLEAN_ROOT_DIR, filetype=CLEAN_OUTPUT_FILETYPE, years_requested=years_requested,
                        filter_fxs=_filter_fxs)
    
    # awaydesc = df['AWAYDESCRIPTION'].dropna()
    # awaytimeouts = awaydesc[awaydesc.str.contains('Time')]
    # timeout_type = df['TIMEOUT_TYPE'].dropna()
    df['remaining_in_quarter'] = pd.to_datetime(df['remaining_in_quarter'])
    last_play_before_2_mins = \
        df[df['remaining_in_quarter'] >= pd.datetime.today().replace(hour=2, minute=0, second=0)].groupby('game_id')[
            'remaining_in_quarter'].last().reset_index()
    last_play_before_2_mins['last_play_before_2_mins'] = True
    
    df = df.merge(last_play_before_2_mins, on=['game_id', 'remaining_in_quarter'], how='left')
    
    df['last_play_before_2_mins'].fillna(False, inplace=True)
    df['SCOREMARGIN'] = df['SCOREMARGIN'].ffill()
    df['SCOREMARGIN'] = df['SCOREMARGIN'].bfill()
    df['SCOREMARGIN'] = df['SCOREMARGIN'].replace('TIE', 0)
    df['SCOREMARGIN'] = df['SCOREMARGIN'].astype(int)
    
    final_home_margin = df.groupby('game_id')['SCOREMARGIN'].last().reset_index()
    final_margin_ix_game = final_home_margin.rename(columns={'SCOREMARGIN': 'final_score_margin'}).set_index('game_id')
    
    two_mins_home_margin = \
        df[df['last_play_before_2_mins']].drop_duplicates(subset='game_id', keep='last').set_index('game_id')[
            'SCOREMARGIN'].rename('two_mins_score_margin')
    
    two_min_outcomes = final_margin_ix_game.join(two_mins_home_margin, how='inner')
    
    two_min_outcomes['winning_team'] = np.where(
        two_min_outcomes['final_score_margin'] > 0,
        'home',
        'away'
    )
    
    two_min_outcomes['winning_team_two_min_margin'] = np.where(
        two_min_outcomes['final_score_margin'] > 0,
        two_min_outcomes['two_mins_score_margin'],
        two_min_outcomes['two_mins_score_margin'] * (-1)
    )
    two_min_outcomes['leading_team_won'] = np.where(
        two_min_outcomes['two_mins_score_margin'] > 0,
        np.where(two_min_outcomes['final_score_margin'] > 0, 1, 0),
        np.where(two_min_outcomes['final_score_margin'] < 0, 1, 0)
    )
    two_min_outcomes['abs_two_mins_score_margin'] = two_min_outcomes['two_mins_score_margin'].abs()
    return two_min_outcomes


# %%

_years_requested = [
    [2008, 2009, 2010, 2011],
    [2013, 2014, 2015],
    [2017, 2018]
]

agg_two_min_outcomes = pd.concat([get_two_minute_outcomes(years_requested=years) for years in _years_requested])

agg_two_min_outcomes.groupby('abs_two_mins_score_margin')['leading_team_won'].mean()
