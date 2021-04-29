# %% imports
import pandas as pd

from nba_pbp_analysis.data.cf import GLOBAL_SAVE_FINAL_FILES_CSV, GLOBAL_SAVE_INTERMEDIATE_FILES_CSV
from nba_pbp_analysis.data.local_io_utils import LocalIOUtils
from nba_pbp_analysis.data.intermediate.base_intm_dataset import BaseIntermediateDataIO


# %% RequestClass

class IntermediateDataRequest:
    save_final_file = GLOBAL_SAVE_FINAL_FILES_CSV
    save_int_files = GLOBAL_SAVE_INTERMEDIATE_FILES_CSV
    
    @classmethod
    def get_data(cls, data_sourcer: BaseIntermediateDataIO, years_requested: list, force_source: bool = False,
                 local_io: LocalIOUtils = None, **kwargs) -> pd.DataFrame:
        if local_io is not None:
            data_sourcer.local_io = local_io
        else:
            local_io = data_sourcer.local_io
        
        if force_source is True:
            return cls.run_source(data_sourcer=data_sourcer, years_requested=years_requested, local_io=local_io,
                                  **kwargs)
        else:
            try:
                op_fileid = "{}_{}_op_{}-{}".format(data_sourcer.version_id, data_sourcer.request_id,
                                                    years_requested[0], years_requested[-1] + 1)
                df_data = local_io.read_csv(fileid=op_fileid)
                return df_data
            except Exception as err:
                print(err)
                return cls.run_source(data_sourcer=data_sourcer, years_requested=years_requested, local_io=local_io,
                                      **kwargs)
    
    @classmethod
    def run_source(cls, data_sourcer: BaseIntermediateDataIO, years_requested: list, local_io: LocalIOUtils = None,
                   **kwargs) -> pd.DataFrame:
        df_op = data_sourcer.source(years_requested=years_requested, save_int_files=cls.save_int_files, **kwargs)
        print(df_op)
        if cls.save_final_file is True and local_io is not None:
            op_fileid = "{}_{}_op_{}-{}".format(data_sourcer.version_id, data_sourcer.request_id, years_requested[0],
                                                years_requested[-1] + 1)
            local_io.save_csv(df=df_op, fileid=op_fileid)
        return df_op
