# %% imports
from nba_pbp_analysis.data.cf import *
from nba_pbp_analysis.data.pbp.cleaning.raw_sources import PBPCleaner834, PBPCleanerBDB


# %% main

def main():
    ### for each of the data sources:
    # 1. init data cleaner object (builds some indexes used in the cleaning process)
    # 2. run data cleaning object's cleaning function
    
    data_src_834 = PBPCleaner834(raw_dir=RAW_ROOT_DIR_834, clean_dir=CLEAN_ROOT_DIR,
                                 output_filetype=CLEAN_OUTPUT_FILETYPE)
    data_src_834.clean_raw_files(
        input_filetype=".csv",
        output_filetype=CLEAN_OUTPUT_FILETYPE,
        overwrite=OVERWRITE_FILES
    )
    
    data_src_BDB = PBPCleanerBDB(raw_dir=RAW_ROOT_DIR_BIGDATABALL, clean_dir=CLEAN_ROOT_DIR,
                                 output_filetype=CLEAN_OUTPUT_FILETYPE)
    data_src_BDB.clean_raw_files(
        input_filetype=".csv",
        output_filetype=CLEAN_OUTPUT_FILETYPE,
        overwrite=OVERWRITE_FILES
    )


if __name__ == '__main__':
    main()
