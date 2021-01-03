# %% imports
import datetime as dt
import os
import pandas as pd
import sys
from collections import OrderedDict

from nba_pbp_analysis.data.cf import COLS_FOR_834_PLAYER_ID_MAPPING_FILEPATH
from nba_pbp_analysis.data.pbp.cleaning.pbp_cleaner import PBPCleaner


# %% globals


# %% classes
class PBPCleaner834(PBPCleaner):
    
    # region setup
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    
    @staticmethod
    def _build_raw_clean_filepath_map(raw_dir: str, clean_dir: str, generic_filename: str = "clean_nba_pbp",
                                      output_filetype: str = ".csv") -> OrderedDict:
        assert os.path.isdir(raw_dir)
        assert os.path.isdir(clean_dir)
        
        allfiles = [f for f in os.listdir(raw_dir) if os.path.isfile(os.path.join(raw_dir, f))]
        pbpfiles = [f for f in allfiles if f[-7:] == "pbp.csv"]
        filename_map = OrderedDict({f: generic_filename + f[6:16] + output_filetype for f in pbpfiles})
        filepath_map = OrderedDict({raw_dir + r: clean_dir + c for r, c in filename_map.items()})
        return filepath_map
    
    # endregion
    
    # region cleaning_functions
    @staticmethod
    def _parse_src_errors(df: pd.DataFrame) -> pd.DataFrame:
        """map source error vals to master error vals"""
        # all good for this source
        return df
    
    @staticmethod
    def _map_player_ids_to_names(df: pd.DataFrame, player_map) -> pd.DataFrame:
        """map player ids to player names"""
        df_cols_for_id_mapping = pd.read_csv(COLS_FOR_834_PLAYER_ID_MAPPING_FILEPATH)
        df_cols_for_id_mapping = df_cols_for_id_mapping[df_cols_for_id_mapping["player_id_col"].isin(df.columns)]
        dict_cols_for_id_mapping = df_cols_for_id_mapping.set_index("player_id_col")["player_name_col"].to_dict()
        n = len(dict_cols_for_id_mapping)
        for i, (id_col, name_col) in enumerate(dict_cols_for_id_mapping.items()):
            df[name_col] = df[id_col].dropna().astype(int).map(player_map)
            print("completed mapping player ids (column {}/{})".format(i+1, n))
        return df
        
    
    @staticmethod
    def _map_src_columns_to_master(src_columns, column_map):
        """rename dataframe columns (map source data columns to master columns)"""
        column_map_dict = \
        column_map[["col_834", "master_col"]].dropna().drop_duplicates("col_834").set_index("col_834")[
            "master_col"].to_dict()
        return [column_map_dict[col] for col in src_columns]
    
    @staticmethod
    def _retype_data(df):
        """parse and re-type data as necessary (to match master spec's)"""
        # TODO need to retype some cols to datetime
        # raise NotImplementedError
        return df
        
    
    
    @classmethod
    def _calc_derived_data(cls, df: pd.DataFrame) -> pd.DataFrame:
        # TODO see notes in the master_column_map file
        return df
    
    # endregion
