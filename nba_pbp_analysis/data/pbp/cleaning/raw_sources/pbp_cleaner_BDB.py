# %% imports
import datetime as dt
import os
import pandas as pd
import sys
from collections import OrderedDict

from nba_pbp_analysis.data.pbp.cleaning.pbp_cleaner import PBPCleaner


# %% classes
class PBPCleanerBDB(PBPCleaner):
    
    # region setup
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    
    @staticmethod
    def _build_raw_clean_filepath_map(raw_dir: str, clean_dir: str, generic_filename: str = "clean_nba_pbp",
                                      output_filetype: str = ".csv") -> OrderedDict:
        assert os.path.isdir(raw_dir)
        assert os.path.isdir(clean_dir)
        
        allsubdirs = [f for f in os.listdir(raw_dir) if os.path.isdir(os.path.join(raw_dir, f))]
        pbpdirs = [os.path.join(raw_dir, f) for f in allsubdirs if f[-8:] == "PbP_Logs"]
        pbpfiles = [i for l in [[os.path.join(d, f) for f in os.listdir(d) if
                                 os.path.isfile(os.path.join(d, f)) and f[-19:] == "-combined-stats.csv"] for d in
                                pbpdirs]
                    for i in l]
        filename_map = OrderedDict(
            {f: "{}_{}-{}{}".format(generic_filename, f[-37:-33], f[-24:-20], output_filetype) for f in pbpfiles})
        filepath_map = OrderedDict({r: clean_dir + c for r, c in filename_map.items()})
        return filepath_map
    
    # endregion
    
    # region cleaning_functions
    @staticmethod
    def _read_file_to_df(filepath: str, filetype: str = ".csv") -> pd.DataFrame:
        """overridden to specify index"""
        if filetype == ".csv":
            return pd.read_csv(filepath)
        elif filetype == ".pkl":
            return pd.read_pickle(filepath)
        else:
            raise NotImplementedError
    
    @staticmethod
    def _parse_src_errors(df: pd.DataFrame) -> pd.DataFrame:
        """map source error vals to master error vals"""
        # all good for this source
        return df
    
    @staticmethod
    def _map_player_ids_to_names(df: pd.DataFrame, player_map) -> pd.DataFrame:
        """map player ids to player names"""
        # all good for this source
        return df
    
    @staticmethod
    def _map_src_columns_to_master(src_columns, column_map):
        """rename dataframe columns (map source data columns to master columns)"""
        column_map_dict = \
            column_map[["col_BDB", "master_col"]].dropna().drop_duplicates("col_BDB").set_index("col_BDB")[
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
