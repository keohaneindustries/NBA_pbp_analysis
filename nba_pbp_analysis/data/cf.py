import pandas as pd

pd.options.display.width = 175
pd.options.display.max_columns = 100
pd.options.display.max_rows = 100

MASTER_COLUMN_MAP_FILEPATH = "C:\\Users\\lukeo\\Documents\\Basketball\\NBA_pbp_analysis\\nba_pbp_analysis\\data\\pbp\\maps\\master_column_map.csv"
MASTER_PLAYER_ID_MAP_FILEPATH = "C:\\Users\\lukeo\\Documents\\Basketball\\NBA_pbp_analysis\\nba_pbp_analysis\\data\\pbp\\maps\\master_player_id_map.csv"
COLS_FOR_834_PLAYER_ID_MAPPING_FILEPATH = "C:\\Users\\lukeo\\Documents\\Basketball\\NBA_pbp_analysis\\nba_pbp_analysis\\data\\pbp\\maps\\834_cols_for_player_id_mapping.csv"

DATA_ROOT_DIR = "C:\\Users\\lukeo\\Documents\\Basketball\\NBA_pbp_data\\"
RAW_ROOT_DIR_834 = DATA_ROOT_DIR + "raw\\EightThirtyFour\\"
RAW_ROOT_DIR_BIGDATABALL = DATA_ROOT_DIR + "raw\\BigDataBall\\"
CLEAN_ROOT_DIR = DATA_ROOT_DIR + "clean\\"
CLEAN_OUTPUT_FILETYPE = ".csv"
OVERWRITE_FILES = False
