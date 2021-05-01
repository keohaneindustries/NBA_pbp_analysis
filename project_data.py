# %% imports
import pandas as pd
import numpy as np

# from nba_pbp_analysis.data.cf import GLOBAL_SAVE_DIRPATH
from nba_pbp_analysis.data.local_io_utils import LocalIOUtils
from nba_pbp_analysis.data.intermediate.intm_dataset_winners import WinnerData
from nba_pbp_analysis.data.intermediate.intm_dataset_closegames import CloseGamesByGameIDData
from nba_pbp_analysis.data.intermediate.intm_dataset_classifier import ClassifierData
from nba_pbp_analysis.data.intermediate.intm_dataset_timeouts_remaining import TimeoutsRemainingData
from nba_pbp_analysis.data.intermediate.intm_data_request import IntermediateDataRequest

# %% globals
_GLOBAL_SAVE_FINAL_FILES_CSV = True
_GLOBAL_SAVE_INTERMEDIATE_FILES_CSV = True


# https://www.espn.com/nba/story/_/id/19974682/nba-board-governors-approves-rule-drop-outs-18-14-per-game

def get_stored_classifier_data() -> pd.DataFrame:
    _years_requested = list(range(2008, 2019))
    local_io = LocalIOUtils
    requester = IntermediateDataRequest
    df_classifier_data = requester.get_data(data_sourcer=ClassifierData(local_io=local_io),
                                            years_requested=_years_requested).set_index('obs_id')
    return df_classifier_data


# %% main
def main():
    _years_requested = list(range(2008, 2019))

    local_io = LocalIOUtils
    # local_io.save_dirpath = GLOBAL_SAVE_DIRPATH

    requester = IntermediateDataRequest
    # requester.save_final_file = _GLOBAL_SAVE_FINAL_FILES_CSV
    # requester.save_int_files = _GLOBAL_SAVE_INTERMEDIATE_FILES_CSV

    df_winners = requester.get_data(data_sourcer=WinnerData(local_io=local_io), years_requested=_years_requested)
    df_close_games = requester.get_data(data_sourcer=CloseGamesByGameIDData(local_io=local_io),
                                        years_requested=_years_requested)
    df_timeouts_remaining = requester.get_data(data_sourcer=TimeoutsRemainingData(local_io=local_io),
                                               years_requested=_years_requested)

    # TODO ANOVA data: index=obs_id
    # TODO get all plays last 3 mins
    # TODO filter to "legit" plays (e.g. not substitutions) - make sure to retain timeouts for next step
    # TODO flag offensive plays immediately following timeouts (append bool; not filter)
    # TODO calc for each play: points scored on play (for team that called timeout)
    # TODO left join close game; filter to close game == True
    #
    df_classifier_data = requester.get_data(data_sourcer=ClassifierData(local_io=local_io),
                                            years_requested=_years_requested, df_winners=df_winners,
                                            df_close_games=df_close_games, df_timeouts_remaining=df_timeouts_remaining)

    return


if __name__ == '__main__':
    main()
