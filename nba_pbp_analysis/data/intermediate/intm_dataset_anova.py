# %% imports
import datetime as dt
import pandas as pd
import numpy as np

from nba_pbp_analysis.data.intermediate.base_intm_dataset import BaseIntermediateDataIO


# %% determine eventual winner (by gameid)

class AnovaData(BaseIntermediateDataIO):
    request_id = "anova_playid"

    @classmethod
    def _filter_fxs(cls, max_rem_in_quarter: dt.datetime, **kwargs) -> list:
        return [
            lambda df: df[df['period'] == 4],
            # lambda df: df[pd.to_datetime(df['remaining_in_quarter']) <= max_remaining_in_quarter]
            lambda df: df[pd.to_datetime(df['remaining_in_quarter'], format="%M:%S") <= max_rem_in_quarter]
        ]

    @classmethod
    def source(cls, years_requested: list, df_winners=None, df_close_games=None, df_timeouts_remaining=None,
               **kwargs) -> pd.DataFrame:
        print("sourcing AnovaData...")
        ## input
        # get all plays last 3 mins
        max_rem_in_quarter = dt.datetime(year=1900, month=1, day=1, hour=0, minute=4, second=0)
        df = cls.read_raw_data(years_requested=years_requested, max_rem_in_quarter=max_rem_in_quarter, **kwargs)
        # _slice_to_relevant_vars_for_anova_calcs
        df = cls._slice_to_relevant_vars_for_anova_calcs(df=df)
        # filter to "legit" plays (e.g. not substitutions) - make sure to retain timeouts for next step
        df = cls._filter_out_trivial_plays(df=df)

        ## calcs
        # infer possession
        df = cls._infer_possession(df=df)
        # calc for each play: points scored on play
        df = cls._calc_points_scored_on_play(df=df)
        # aggregate consecutive offensive plays
        df = cls._aggregate_consecutive_offensive_plays(df=df)
        # flag offensive plays immediately following timeouts (append bool; not filter)
        df = cls._flag_offensive_plays_immediately_following_timeout(df=df)

        ## merging datasets
        # left join close game; filter to close game == True
        df = df.merge(df_close_games, how='left', on='game_id')
        df = cls._filter_to_close_games(df=df)

        ## removing extraneous/duplicate data
        df = cls._remove_timeout_plays(df=df)
        max_rem_in_quarter = dt.datetime(year=1900, month=1, day=1, hour=0, minute=3, second=0)
        df = cls._slice_to_relevant_timespan(df=df, max_rem_in_quarter=max_rem_in_quarter)
        df = cls._slice_to_relevant_vars_for_home_away_split(df=df)  # TODO

        ## transform home/away
        df = cls._transform_home_vs_away(df=df)

        ## output
        # slice to output vars
        df = cls._slice_to_output_vars(df=df)
        df.reset_index(drop=True, inplace=True)
        df.index.name = "obs_id"
        return df

    @staticmethod
    def _slice_to_relevant_vars_for_anova_calcs(df: pd.DataFrame) -> pd.DataFrame:
        relevant_vars = ['game_id', 'play_id', 'event_subtype_id', 'event_type_id', 'HOMEDESCRIPTION',
                         'AWAYDESCRIPTION', 'home_score', 'away_score', 'rem_in_quarter_dt'
                         ]
        return df.loc[:, relevant_vars]

    @classmethod
    def _filter_out_trivial_plays(cls, df: pd.DataFrame) -> pd.DataFrame:
        # drop duplicates
        df = df.drop_duplicates()

        # subtype_codes = {
        #     1: "made_shot",
        #     2: "missed_shot",
        #     3: "free_throw attempt (some made, some missed)",
        #     4: "rebound",
        #     5: "turnover",
        #     6: {2: "shooting_foul", 4: "offensive_foul"},
        #     7: {1: "delay_of_game", 2: "goaltending", 3: "lane_violation", 4: "kicked_ball"},
        #     8: "substitution",
        #     9: "timeout",
        #     10: "jump_ball",
        #     11: "ejection",
        #     13: np.nan,
        #     18: np.nan
        # }
        good_subtype_ids = [1, 2, 3, 5, 9]
        df_good = df[np.where(
            df['event_subtype_id'].isin(good_subtype_ids),
            True,
            np.where(
                (df['event_subtype_id'] == 6) & (df['event_type_id'] == 4),
                True,
                np.where(
                    (df['event_subtype_id'] == 7) & (df['event_type_id'].isin([1, 2])),
                    True,
                    False
                )
            )
        )]
        return df_good

    @classmethod
    def _infer_possession(cls, df: pd.DataFrame) -> pd.DataFrame:
        home_shot = (~df['HOMEDESCRIPTION'].isna()) & (df['event_subtype_id'].isin([1, 2, 3]))
        away_shot = (~df['AWAYDESCRIPTION'].isna()) & (df['event_subtype_id'].isin([1, 2, 3]))

        home_turnover = df['HOMEDESCRIPTION'].str.contains("Turnover").fillna(False)
        away_turnover = df['AWAYDESCRIPTION'].str.contains("Turnover").fillna(False)

        home_offensive_foul = (~df['HOMEDESCRIPTION'].isna()) & (df['event_subtype_id'] == 6) & (
                df['event_type_id'] == 4)
        away_offensive_foul = (~df['AWAYDESCRIPTION'].isna()) & (df['event_subtype_id'] == 6) & (
                df['event_type_id'] == 4)

        home_delay_of_game = (~df['HOMEDESCRIPTION'].isna()) & (df['event_subtype_id'] == 7) & (
                df['event_type_id'] == 1)
        away_delay_of_game = (~df['AWAYDESCRIPTION'].isna()) & (df['event_subtype_id'] == 7) & (
                df['event_type_id'] == 1)

        home_goaltend = (~df['HOMEDESCRIPTION'].isna()) & (df['event_subtype_id'] == 7) & (df['event_type_id'] == 2)
        away_goaltend = (~df['AWAYDESCRIPTION'].isna()) & (df['event_subtype_id'] == 7) & (df['event_type_id'] == 2)

        home_timeout = (~df['HOMEDESCRIPTION'].isna()) & (df['event_subtype_id'] == 9)
        away_timeout = (~df['AWAYDESCRIPTION'].isna()) & (df['event_subtype_id'] == 9)

        home_possession = home_shot | home_turnover | home_offensive_foul | home_delay_of_game | away_goaltend
        away_possession = away_shot | away_turnover | away_offensive_foul | away_delay_of_game | home_goaltend

        df['possession'] = np.where(
            home_timeout, "home_timeout",
            np.where(
                away_timeout, "away_timeout",
                np.where(
                    home_possession, "home",
                    np.where(
                        away_possession, "away", None
                    )
                )
            )

        )
        df = df[~df['possession'].isna()]

        return df

    @staticmethod
    def _calc_points_scored_on_play(df: pd.DataFrame) -> pd.DataFrame:
        df['home_score'] = df['home_score'].astype(np.float)
        df['away_score'] = df['away_score'].astype(np.float)
        df["home_points_on_play"] = df.groupby('game_id')['home_score'].diff()
        df["away_points_on_play"] = df.groupby('game_id')['away_score'].diff()
        return df

    @staticmethod
    def _aggregate_consecutive_offensive_plays(df: pd.DataFrame) -> pd.DataFrame:
        agg = (df['possession'].shift() != df['possession']).cumsum()
        pop = df.groupby(agg)[['home_points_on_play', 'away_points_on_play']].sum().reset_index(drop=True)

        cols_to_keep = ['game_id', 'play_id', 'event_type_id', 'event_subtype_id', 'possession', 'rem_in_quarter_dt']
        df_descriptive = df.groupby(agg)[cols_to_keep].last().reset_index(drop=True)

        df_agg = pd.concat([pop, df_descriptive], axis=1)
        return df_agg

    @classmethod
    def _flag_offensive_plays_immediately_following_timeout(cls, df: pd.DataFrame) -> pd.DataFrame:
        df['home_poss_follow_to'] = (df['possession'].shift(-1) == "home_timeout") & (df['possession'] == "home") & (
                df['game_id'].shift(-1) == df['game_id'])
        df['away_poss_follow_to'] = (df['possession'].shift(-1) == "away_timeout") & (df['possession'] == "away") & (
                df['game_id'].shift(-1) == df['game_id'])
        return df

    @staticmethod
    def _filter_to_close_games(df: pd.DataFrame) -> pd.DataFrame:
        return df[df['close_game']]

    @staticmethod
    def _remove_timeout_plays(df: pd.DataFrame) -> pd.DataFrame:
        return df[df['event_subtype_id'] != 9]

    @staticmethod
    def _slice_to_relevant_timespan(df: pd.DataFrame, max_rem_in_quarter: dt.datetime) -> pd.DataFrame:
        one_s = dt.datetime(year=1900, month=1, day=1, hour=0, minute=0, second=1)
        return df[(df['rem_in_quarter_dt'] <= max_rem_in_quarter) & (df['rem_in_quarter_dt'] >= one_s)].dropna()

    @staticmethod
    def _slice_to_relevant_vars_for_home_away_split(df: pd.DataFrame) -> pd.DataFrame:
        relevant_vars = ['game_id', 'play_id', 'possession', 'home_poss_follow_to', 'away_poss_follow_to',
                         'home_points_on_play', 'away_points_on_play']
        return df.loc[:, relevant_vars]

    @classmethod
    def _transform_home_vs_away(cls, df: pd.DataFrame) -> pd.DataFrame:
        df['is_home'] = np.where(df['possession'] == "home", True, False)
        df['points_on_play'] = np.where(df['possession'] == "home", df['home_points_on_play'],
                                        df['away_points_on_play'])
        df['poss_follow_to'] = np.where(df['possession'] == "home", df['home_poss_follow_to'],
                                        df['away_poss_follow_to'])
        return df

    @staticmethod
    def _slice_to_output_vars(df: pd.DataFrame) -> pd.DataFrame:
        relevant_vars = ['game_id', 'play_id', 'is_home', 'poss_follow_to', 'points_on_play']
        return df.loc[:, relevant_vars]
