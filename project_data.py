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
    
    return


if __name__ == '__main__':
    main()
