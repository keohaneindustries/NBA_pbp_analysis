# %% imports
from abc import abstractmethod
import datetime as dt
import numpy as np
import os
import pandas as pd
import sys
from collections import OrderedDict

from nba_pbp_analysis.data.cf import MASTER_COLUMN_MAP_FILEPATH, MASTER_PLAYER_ID_MAP_FILEPATH


class PBPCleaner:
    
    # region setup
    def __init__(self, **kwargs):
        self.raw_to_clean_filepath_map = self._build_raw_clean_filepath_map(**kwargs)
        self.player_map = self._build_player_map()
        self.column_map = self._build_column_map()
    
    @staticmethod
    @abstractmethod
    def _build_raw_clean_filepath_map(raw_dir: str, clean_dir: str, generic_filename: str = "clean_nba_pbp",
                                      output_filetype: str = ".csv") -> OrderedDict:
        raise NotImplementedError
    
    def _build_player_map(self):
        df_player_map = pd.read_csv(MASTER_PLAYER_ID_MAP_FILEPATH, index_col=0)
        return df_player_map.set_index("PERSON_ID")["DISPLAY_FIRST_LAST"].to_dict()
    
    def _build_column_map(self):
        df_column_map = pd.read_csv(MASTER_COLUMN_MAP_FILEPATH)
        df_column_map.loc[:, ['col_BDB', 'col_834']] = df_column_map.loc[:, ['col_BDB', 'col_834']].replace("[X]",
                                                                                                            np.nan)
        df_column_map.loc[df_column_map['col_834'] == '[map player id]', 'col_834'] = df_column_map.loc[
            df_column_map['col_834'] == '[map player id]', 'master_col']
        return df_column_map
    
    # endregion
    
    # region cleaning
    def clean_raw_files(self, input_filetype: str = ".csv", output_filetype: str = ".csv",
                        overwrite: bool = True) -> None:
        for (raw_filepath, clean_filepath) in self.raw_to_clean_filepath_map.items():
            if (overwrite is not True) and (os.path.exists(clean_filepath)):
                print("skipping overwrite; file exists: {}".format(clean_filepath))
            else:
                self._read_clean_save(player_map=self.player_map, column_map=self.column_map, raw_filepath=raw_filepath,
                                      input_filetype=input_filetype, clean_filepath=clean_filepath,
                                      output_filetype=output_filetype)
    
    @classmethod
    def _read_clean_save(cls, player_map, column_map, raw_filepath: str, clean_filepath: str,
                         input_filetype: str = ".csv", output_filetype: str = ".csv"):
        # this is a function so that the df gets removed from the cache between loops for better performance
        print("starting cleaning for file {}...".format(raw_filepath))
        df = cls._read_file_to_df(filepath=raw_filepath, filetype=input_filetype)
        df = cls._clean_data(df=df, player_map=player_map, column_map=column_map)
        cls._save_data_to_file(df, filepath=clean_filepath, filetype=output_filetype)
    
    @staticmethod
    def _read_file_to_df(filepath: str, filetype: str = ".csv") -> pd.DataFrame:
        """read file to dataframe. method can be overridden to specify index, chunking, etc."""
        if filetype == ".csv":
            return pd.read_csv(filepath, index_col=0)
        elif filetype == ".pkl":
            return pd.read_pickle(filepath)
        else:
            raise NotImplementedError
    
    # region cleaning_functions
    @classmethod
    def _clean_data(cls, df: pd.DataFrame, player_map, column_map) -> pd.DataFrame:
        df.reset_index(drop=True, inplace=True)
        df = cls._parse_src_errors(df)
        df = cls._map_player_ids_to_names(df, player_map=player_map)
        df.columns = cls._map_src_columns_to_master(src_columns=df.columns, column_map=column_map)
        df = cls._retype_data(df)
        df = cls._calc_derived_data(df)
        return df
    
    @staticmethod
    @abstractmethod
    def _parse_src_errors(df: pd.DataFrame) -> pd.DataFrame:
        """map source error vals to master error vals"""
        raise NotImplementedError
    
    @staticmethod
    @abstractmethod
    def _map_player_ids_to_names(df: pd.DataFrame, player_map) -> pd.DataFrame:
        """map player ids to player names"""
        raise NotImplementedError
    
    @staticmethod
    @abstractmethod
    def _map_src_columns_to_master(src_columns, column_map):
        """rename dataframe columns (map source data columns to master columns)"""
        raise NotImplementedError
    
    @staticmethod
    @abstractmethod
    def _retype_data(df):
        """parse and re-type data as necessary (to match master spec's)"""
        raise NotImplementedError
    
    @classmethod
    def _calc_derived_data(cls, df: pd.DataFrame) -> pd.DataFrame:
        return df
    
    # endregion
    
    @staticmethod
    def _save_data_to_file(df: pd.DataFrame, filepath: str, filetype: str = ".csv") -> None:
        if filetype == ".csv":
            df.to_csv(filepath)
            print("saved df ({}) to {}".format(df.shape, filepath))
        elif filetype == ".pkl":
            df.to_pickle(filepath)
            print("saved df ({}) to {}".format(df.shape, filepath))
        else:
            raise NotImplementedError
    # endregion
