# %% imports
import os
import pandas as pd

from nba_pbp_analysis.data.cf import GLOBAL_SAVE_DIRPATH


# %% class
class LocalIOUtils:
    save_dirpath = GLOBAL_SAVE_DIRPATH
    
    @classmethod
    def read_csv(cls, filepath: str = None, fileid: str = None, **kwargs) -> pd.DataFrame:
        filepath = filepath if filepath is not None else cls.filepath(fileid=fileid)
        df = pd.read_csv(filepath, **kwargs)
        print("read df {} from ... {}".format(df.shape, filepath[-(min([50, len(filepath)])):]))
        return df
    
    @classmethod
    def save_csv(cls, df: pd.DataFrame, filepath: str = None, fileid: str = None, **kwargs) -> None:
        filepath = filepath if filepath is not None else cls.filepath(fileid=fileid)
        df.to_csv(filepath, **kwargs)
        print("saved df {} to ... {}".format(df.shape, filepath[-(min([50, len(filepath)])):]))
        return
    
    @classmethod
    def filepath(cls, fileid: str = None) -> str:
        return os.path.join(cls.save_dirpath, "{}.csv".format(fileid))
