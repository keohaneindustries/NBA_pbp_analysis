# %% imports
from abc import abstractmethod
import datetime as dt
import os
import pandas as pd
import numpy as np

from nba_pbp_analysis.data.cf import GLOBAL_VERSION_ID, CLEAN_ROOT_DIR, CLEAN_OUTPUT_FILETYPE
from nba_pbp_analysis.data.load import load_clean_pbp
from nba_pbp_analysis.data.local_io_utils import LocalIOUtils


# %% base class for intermediate data IO
class BaseIntermediateDataIO:
    version_id = GLOBAL_VERSION_ID
    request_id = None
    local_io = LocalIOUtils
    
    def __init__(self, local_io: LocalIOUtils = None):
        if local_io is not None:
            self.local_io = local_io
    
    @classmethod
    def source(cls, years_requested: list, save_int_files: bool = True, **kwargs) -> pd.DataFrame:
        return cls.loop_thru_years(years_requested=years_requested, save_int_files=save_int_files, **kwargs)
    
    @classmethod
    def loop_thru_years(cls, years_requested: list, save_int_files: bool = True, **kwargs) -> pd.DataFrame:
        year_stripes = cls._stripe_years(years_requested=years_requested)
        l_slices = []
        for stripe in year_stripes:
            df_year_slice = cls.source(years_requested=stripe, **kwargs)
            if save_int_files is True:
                year_fileid = "{}_{}_year_{}-{}".format(cls.version_id, cls.request_id, stripe[0], stripe[-1] + 1)
                cls.local_io.save_csv(df=df_year_slice, fileid=year_fileid)
            l_slices.append(df_year_slice.copy())
        return pd.concat(l_slices)
    
    @staticmethod
    def _stripe_years(years_requested: list, max_n: int = 2) -> list:
        n = len(years_requested)
        mod = n % max_n
        
        l_stripes = [[]]
        for i, year in enumerate(years_requested):
            x = l_stripes[-1]
            if (i + mod) > n:
                x.append(year)
            elif len(x) < (max_n - 1):
                x.append(year)
            else:
                l_stripes.append([])
        return l_stripes
    
    @classmethod
    @abstractmethod
    def source(cls, **kwargs) -> pd.DataFrame:
        raise NotImplementedError
    
    @classmethod
    def read_raw_data(cls, years_requested: list, **kwargs) -> pd.DataFrame:
        return load_clean_pbp(clean_dir=CLEAN_ROOT_DIR, filetype=CLEAN_OUTPUT_FILETYPE, years_requested=years_requested,
                              filter_fxs=cls._filter_fxs(**kwargs))
    
    @classmethod
    @abstractmethod
    def _filter_fxs(cls, **kwargs) -> list:
        raise NotImplementedError
