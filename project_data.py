# %% imports
import pandas as pd
import numpy as np

# from nba_pbp_analysis.data.cf import GLOBAL_SAVE_DIRPATH
from nba_pbp_analysis.data.local_io_utils import LocalIOUtils
from nba_pbp_analysis.data.intermediate.intm_dataset_anova import AnovaData
from nba_pbp_analysis.data.intermediate.intm_dataset_winners import WinnerData
from nba_pbp_analysis.data.intermediate.intm_dataset_closegames import CloseGamesByGameIDData
from nba_pbp_analysis.data.intermediate.intm_dataset_classifier import ClassifierData
from nba_pbp_analysis.data.intermediate.intm_dataset_timeouts_remaining import TimeoutsRemainingData
from nba_pbp_analysis.data.intermediate.intm_data_request import IntermediateDataRequest

# %% globals
_GLOBAL_SAVE_FINAL_FILES_CSV = True
_GLOBAL_SAVE_INTERMEDIATE_FILES_CSV = True


# https://www.espn.com/nba/story/_/id/19974682/nba-board-governors-approves-rule-drop-outs-18-14-per-game

def get_stored_anova_data() -> pd.DataFrame:
    _years_requested = list(range(2008, 2019))
    local_io = LocalIOUtils
    requester = IntermediateDataRequest
    df_anova_data = requester.get_data(data_sourcer=AnovaData(local_io=local_io),
                                       years_requested=_years_requested).set_index('obs_id')
    return df_anova_data


def get_stored_classifier_data() -> pd.DataFrame:
    _years_requested = list(range(2008, 2019))
    local_io = LocalIOUtils
    requester = IntermediateDataRequest
    df_classifier_data = requester.get_data(data_sourcer=ClassifierData(local_io=local_io),
                                            years_requested=_years_requested).set_index('obs_id')
    return df_classifier_data


def convert_stored_classifier_data_to_predictors(df: pd.DataFrame) -> pd.DataFrame:
    # check that the stored data contains the columns we think it does
    req_col_names = ['did_win_game', 't_remaining_s', 'game_points_per_min', 'rel_score_margin', 'team_rem_timeouts',
                     'opp_rem_timeouts']
    for col in req_col_names:
        assert col in df.columns

    df = df[df['t_remaining_s'] > 0.]  # drop observations where the game has ended

    # define interaction variables
    df.loc[:, 'point_margin_per_min_rem'] = df['rel_score_margin'] / df['t_remaining_s'] * 60.
    df.loc[:, 'std_point_margin_per_min_rem'] = df['point_margin_per_min_rem'] / df['game_points_per_min'] * df[
        'game_points_per_min'].mean()
    return df


def drop_extraneous_predictors(df: pd.DataFrame) -> pd.DataFrame:
    cols_to_drop = [
        't_remaining_s',
        'game_points_per_min',
        'rel_score_margin',
        'opp_rem_timeouts',
        'point_margin_per_min_rem',
        # 'std_point_margin_per_min_rem'
    ]
    return df.drop(cols_to_drop, axis=1)


# %% main
def main():
    _years_requested = list(range(2008, 2019))
    # _years_requested = list(range(2008, 2009))

    local_io = LocalIOUtils
    # local_io.save_dirpath = GLOBAL_SAVE_DIRPATH

    requester = IntermediateDataRequest
    # requester.save_final_file = _GLOBAL_SAVE_FINAL_FILES_CSV
    # requester.save_int_files = _GLOBAL_SAVE_INTERMEDIATE_FILES_CSV

    df_close_games = requester.get_data(data_sourcer=CloseGamesByGameIDData(local_io=local_io),
                                        years_requested=_years_requested)

    # ANOVA data: index=obs_id
    df_anova_data = requester.get_data(data_sourcer=AnovaData(local_io=local_io),
                                       years_requested=_years_requested, df_close_games=df_close_games)

    df_winners = requester.get_data(data_sourcer=WinnerData(local_io=local_io), years_requested=_years_requested)
    df_timeouts_remaining = requester.get_data(data_sourcer=TimeoutsRemainingData(local_io=local_io),
                                               years_requested=_years_requested)

    #
    df_classifier_data = requester.get_data(data_sourcer=ClassifierData(local_io=local_io),
                                            years_requested=_years_requested, df_winners=df_winners,
                                            df_close_games=df_close_games, df_timeouts_remaining=df_timeouts_remaining)

    return


if __name__ == '__main__':
    main()
