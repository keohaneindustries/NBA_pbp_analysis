# %% imports
import pandas as pd
import numpy as np

# from nba_pbp_analysis.data.cf import GLOBAL_SAVE_DIRPATH
from nba_pbp_analysis.data.local_io_utils import LocalIOUtils
from nba_pbp_analysis.data.intermediate.intm_dataset_winners import WinnerData
from nba_pbp_analysis.data.intermediate.intm_dataset_closegames import CloseGamesByGameIDData
from nba_pbp_analysis.data.intermediate.intm_data_request import IntermediateDataRequest

# %% globals
_GLOBAL_SAVE_FINAL_FILES_CSV = True
_GLOBAL_SAVE_INTERMEDIATE_FILES_CSV = True


# %% main
def main():
    # _years_requested = list(range(2008, 2019))
    # _years_requested = list(range(2008, 2009))
    _years_requested = list(range(2008, 2011))
    
    local_io = LocalIOUtils
    # local_io.save_dirpath = GLOBAL_SAVE_DIRPATH
    
    requester = IntermediateDataRequest
    # requester.save_final_file = _GLOBAL_SAVE_FINAL_FILES_CSV
    # requester.save_int_files = _GLOBAL_SAVE_INTERMEDIATE_FILES_CSV
    
    df_winners = requester.get_data(data_sourcer=WinnerData(local_io=local_io), years_requested=_years_requested)
    df_close_games = requester.get_data(data_sourcer=CloseGamesByGameIDData(local_io=local_io),
                                        years_requested=_years_requested)
    
    # ANOVA data: index=playid
    # TODO get all plays last 3 mins
    # TODO filter to "legit" plays (e.g. not substitutions) - make sure to retain timeouts for next step
    # TODO flag offensive plays immediately following timeouts (append bool; not filter)
    # TODO calc for each play: points scored on play (for team that called timeout)
    # TODO left join close game; filter to close game == True
    
    # Classifier data: index=playid
    # TODO get all plays last 3 mins
    # TODO calc for each play: point margin, time remaining (s), num of timeouts remaining (home, away)
    # TODO drop duplicates
    # TODO left join close game; filter to close game == True
    # TODO left join winners;
    # TODO split into home/away, apply any necessary transformations to align variables, then stack
    # TODO calc for each play "did eventually win game" (bool)
    
    return


if __name__ == '__main__':
    main()
