import numpy as np
import pandas as pd
import os
import re

from sklearn.base import BaseEstimator, TransformerMixin
from datetime import datetime


def create_uid(base_str):
    """
    Creates a UID by concatenating the current timestamp with a base string.

    :param base_str: string to concat the time to
    :returns: str
    """
    now = datetime.now()
    uid = base_str + '_' + str(now).replace(' ', '').replace('.', '').replace(':', '').replace('-', '')
    return uid


def convert_camel_case_to_snake_case(df):
    """
    Converts dataframe column headers from snakeCase to camel_case.

    :param df: any valid pandas dataframe
    :return: pandas dataframe
    """
    new_columns = []
    for column in list(df):
        new_column = re.sub(r'(?<!^)(?=[A-Z])', '_', column).lower()
        new_columns.append(new_column)
    df.columns = new_columns
    return df


def make_directories_if_not_exists(directories_list):
    """
    Makes directories if they do not exist.

    :param directories_list: list of directories to create
    """
    for directory in directories_list:
        if not os.path.exists(directory):
            os.makedirs(directory)


def clean_batting_and_pitching_players(batting_df, pitching_df, positions_df):
    """
    Removes players from batting_df who are pitchers. Likewise, removes players from the pitching_df who are not
    pitchers. This is all based on the information in positions_df.

    :param batting_df: dataframe of batting stats
    :param pitching_df: dataframe of pitching stats
    :param positions_df: dataframe of the positions played by a player
    :returns: batting_df, pitching_df
    """
    positions_df['pitching_percentage'] = positions_df['G_p'] / positions_df['G_all']
    pitchers_df = positions_df.loc[positions_df['pitching_percentage'] >= 0.99]
    pitchers_df['remove_id'] = pitchers_df['playerID'] + pitchers_df['yearID'].astype(str)
    pitchers_ids = pitchers_df['remove_id'].tolist()
    remove_pitchers_df = positions_df.loc[(positions_df['pitching_percentage'] <= 0.02) &
                                          (positions_df['pitching_percentage'] > 0)]
    remove_pitchers_df['remove_id'] = remove_pitchers_df['playerID'] + remove_pitchers_df['yearID'].astype(str)
    remove_pitchers_ids = remove_pitchers_df['remove_id'].tolist()

    batting_df = batting_df.loc[~(batting_df['playerID'] + batting_df['yearID'].astype(str)).isin(pitchers_ids)]
    pitching_df = pitching_df.loc[~(pitching_df['playerID'] + pitching_df['yearID'].astype(str)).isin(
        remove_pitchers_ids)]

    return batting_df, pitching_df


def calculate_expanding_obp(df):
    """
    Calculates an expanding on base percentage.

    :param df: pandas dataframe of raw batting stats
    :returns: pandas dataframe
    """
    coerce_numeric_list = ['H', 'BB', 'HBP', 'AB', 'SF', '2B', '3B', 'HR', 'IBB', 'G']
    for variable in coerce_numeric_list:
        df[variable] = pd.to_numeric(df[variable], errors='coerce')

    features = list(df)
    features.remove('playerID')
    features.remove('yearID')
    features.remove('teamID')

    for feature in features:
        df[feature] = df.groupby('playerID')[feature].shift()

    df['expanding_games'] = df.groupby('playerID')['G'].cumsum()
    df['expanding_hits'] = df.groupby('playerID')['H'].cumsum()
    df['expanding_walks'] = df.groupby('playerID')['BB'].cumsum()
    df['expanding_hbp'] = df.groupby('playerID')['HBP'].cumsum()
    df['expanding_at_bats'] = df.groupby('playerID')['AB'].cumsum()
    df['expanding_sac_flies'] = df.groupby('playerID')['SF'].cumsum()
    df['expanding_doubles'] = df.groupby('playerID')['2B'].cumsum()
    df['expanding_triples'] = df.groupby('playerID')['3B'].cumsum()
    df['expanding_home_runs'] = df.groupby('playerID')['HR'].cumsum()
    df['expanding_intentional_walks'] = df.groupby('playerID')['IBB'].cumsum()

    df['obp'] = (df['expanding_hits'] + df['expanding_walks'] + df['expanding_hbp']) / \
                (df['expanding_at_bats'] + df['expanding_walks'] + df['expanding_hbp'] + df['expanding_sac_flies'])
    df['slg'] = (df['expanding_hits'] - (df['expanding_doubles'] + df['expanding_triples'] + df['expanding_home_runs'])
                 + df['expanding_doubles'] * 2 + df['expanding_triples'] * 3 + df['expanding_home_runs'] * 4) / \
                df['expanding_at_bats']

    df['slg'].fillna(value=0, inplace=True)
    df['obp'].fillna(value=0, inplace=True)
    df['ops'] = df['obp'] + df['slg']
    return df


def calculate_expanding_era(df):
    """
    Calculates expanding earned run average.

    :param df: pandas dataframe with pitching stats
    :returns: dataframe
    """
    df.drop('ERA', 1, inplace=True)
    df['ER'] = df.groupby('playerID')['ER'].shift()
    df['IPouts'] = df.groupby('playerID')['IPouts'].shift()
    df['innings_pitched'] = df['IPouts'] / 3
    df['expanding_earned_runs'] = df.groupby('playerID')['ER'].cumsum()
    df['expanding_innings_pitched'] = df.groupby('playerID')['innings_pitched'].cumsum()
    df['era'] = 9 * df['expanding_earned_runs'] / df['expanding_innings_pitched']
    df['era'] = np.where(df['era'] == 0, np.nan, df['era'])
    return df


def find_total_and_lagged_all_star_appearances(df, year_lags):
    """
    Finds the total number of all-star appearances in a player's career. Also finds the total number of all-star
    appearances for every year until the end of a lookback window, defined by year_lags.

    :param df: pandas dataframe including all-star appearances
    :param year_lags: number of years to lookback for calculating all-star appearances for a window calculation
    :returns: pandas dataframe
    """
    df.dropna(subset=['yearID'], inplace=True)
    df['yearID'] = df['yearID'].astype(int)
    df_2020 = df.loc[df['yearID'] == 2019]
    df_2020['yearID'] = 2020
    df = pd.concat([df, df_2020], axis=0)
    df['all_star'] = 1

    years_df = pd.DataFrame({'yearID': range(df['yearID'].min(), df['yearID'].max() + 2)})
    players = list(df['playerID'].unique())
    main_players_df = pd.DataFrame()
    for player in players:
        player_df = df.loc[df['playerID'] == player]
        player_df = pd.merge(years_df, player_df, how='left', on='yearID')
        lag_cols = []
        for lag in range(1, year_lags + 1):
            col = f'year_lag_{lag}'
            player_df[col] = player_df['all_star'].shift(lag)
            player_df[col] = np.where(player_df[col] > 0, 1, 0)
            lag_cols.append(col)
        appearance_cols = []
        for index, lag_col in enumerate(lag_cols):
            appearance_col = f'as_appearance_last_{lag_col}_year'
            player_df[appearance_col] = player_df[lag_cols[:index + 1]].sum(axis=1)
            appearance_cols.append(appearance_col)
        player_df = player_df[['playerID', 'yearID'] + appearance_cols]
        player_df.fillna({'playerID': player}, inplace=True)
        player_df.dropna(inplace=True)
        main_players_df = main_players_df.append(player_df)

    df = pd.merge(main_players_df, df, how='left', on=['playerID', 'yearID'])
    df.fillna({'all_star': 0}, inplace=True)
    df['expanding_as_appearances'] = df.groupby('playerID')['all_star'].cumsum()
    df['expanding_as_appearances'] = df.groupby('playerID')['expanding_as_appearances'].shift(1)
    df = df.drop('teamID', 1)
    return df


def find_player_age(player_df, stats_df):
    """
    Finds a players age.

    :param player_df: dataframe with birth information
    :param stats_df: dataframe with player stats for every given year
    :returns: pandas dataframe
    """
    player_df = player_df.dropna(subset=['birthYear', 'birthMonth', 'birthDay'])
    player_df['birth_date'] = player_df['birthYear'].astype(int).astype(str) + '-' + \
                              player_df['birthMonth'].astype(int).astype(str) + '-' + \
                              player_df['birthDay'].astype(int).astype(str)
    player_df['birth_date'] = pd.to_datetime(player_df['birth_date'])
    player_df = player_df[['playerID', 'birth_date']]
    stats_df = pd.merge(stats_df, player_df, how='left', on='playerID')
    stats_df['stock_start_date'] = stats_df['yearID'].astype(str) + '-04-01'
    stats_df['stock_start_date'] = pd.to_datetime(stats_df['stock_start_date'])
    stats_df['player_age'] = ((stats_df['stock_start_date'] - stats_df['birth_date']) / np.timedelta64(1, 'Y'))
    return stats_df


def find_yearly_team_winning_percentages(df, year_lags):
    """
    Finds the team's winning percentage every year until the end of a lookback window, defined by year_lags. Also
    calculates the rolling 3 year and 5 year winning percentages.

    :param df: pandas dataframe of team wins and losses
    :param year_lags:: number of years to lookback for calculating winning percentages for a window calculation
    :returns: pandas dataframe
    """
    df['winning_percentage'] = df['W'] / (df['W'] + df['L'])
    for lag in range(1, year_lags + 1):
        df[f'winning_percentage_lag_{lag}'] = df.groupby('teamID')['winning_percentage'].shift(lag)
    df['winning_percentage_rolling_3'] = (df[f'winning_percentage_lag_1'] + df[f'winning_percentage_lag_2'] +
                                         df[f'winning_percentage_lag_3']) / 3
    df['winning_percentage_rolling_5'] = (df[f'winning_percentage_lag_1'] + df[f'winning_percentage_lag_2'] +
                                         df[f'winning_percentage_lag_3'] + df[f'winning_percentage_lag_4'] +
                                          df[f'winning_percentage_lag_5']) / 5
    return df


def find_postseason_results(df, year_lags):
    """
    Finds the team's postseason every year until the end of a lookback window, defined by year_lags.

    :param df: pandas dataframe of team postseason results
    :param year_lags:: number of years to lookback for calculating winning percentages for a window calculation
    :returns: pandas dataframe
    """
    winner_df = df[['yearID', 'round', 'teamIDwinner']]
    winner_df['round'] = winner_df['round'] + ' - winner'
    winner_df.rename(columns={'teamIDwinner': 'teamID'}, inplace=True)

    loser_df = df[['yearID', 'round', 'teamIDloser']]
    loser_df['round'] = loser_df['round'] + ' - loser'
    loser_df.rename(columns={'teamIDloser': 'teamID'}, inplace=True)
    df = pd.concat([winner_df, loser_df], axis=0)

    years_df = pd.DataFrame({'yearID': range(df['yearID'].min(), df['yearID'].max() + 2)})
    teams = list(df['teamID'].unique())
    main_teams_df = pd.DataFrame()
    for team in teams:
        team_df = df.loc[df['teamID'] == team]
        team_df.drop_duplicates(subset=['teamID', 'yearID'], keep='last', inplace=True)
        team_df = pd.merge(years_df, team_df, how='left', on='yearID')
        lag_cols = []
        for lag in range(1, year_lags + 1):
            col = f'year_lag_{lag}'
            team_df[col] = team_df['round'].shift(lag)
            lag_cols.append(col)
        result_cols = []
        for index, lag_col in enumerate(lag_cols):
            appearance_col = f'postseason_result_last_{lag_col}_year'
            team_df[appearance_col] = team_df[lag_cols[index]]
            result_cols.append(appearance_col)
        team_df = team_df[['teamID', 'yearID'] + result_cols]
        team_df.fillna({'teamID': team}, inplace=True)
        team_df.fillna('missed_playoffs', inplace=True)
        main_teams_df = main_teams_df.append(team_df)

    return main_teams_df


def create_target_dataframe(df):
    """
    Creates a dataframe of our target: the team who won the World Series.

    :param df: pandas dataframe including the team who won the World Series
    :returns: pandas dataframe
    """
    ws_df = df.loc[df['round'] == 'WS']
    ws_df = ws_df[['yearID', 'teamIDwinner']]
    ws_df['target'] = 1
    return ws_df


def flatten_data_to_team_and_year(df):
    """
    Flattens the dataframe to be one row for every tear amd year.

    :param df: pandas dataframe
    :returns: pandas dataframe
    """
    aggregates_df = df.groupby(['team_yearID', 'team_teamID']).agg({
        'batting_ops': 'median',
        'pitching_era': 'median',
        'batting_as_appearance_last_year_lag_1_year': 'sum',
        'batting_as_appearance_last_year_lag_3_year': 'sum',
        'batting_expanding_as_appearances': 'sum',
        'pitching_as_appearance_last_year_lag_1_year': 'sum',
        'pitching_as_appearance_last_year_lag_3_year': 'sum',
        'pitching_expanding_as_appearances': 'sum',
        'batting_player_age': 'mean',
        'pitching_player_age': 'mean'
    })
    aggregates_df['total_as_appearances'] = aggregates_df['batting_expanding_as_appearances'] + \
                                            aggregates_df['pitching_expanding_as_appearances']
    df.drop(list(aggregates_df), 1, inplace=True, errors='ignore')
    df.drop_duplicates(subset=['team_yearID', 'team_teamID'], keep='first', inplace=True)
    aggregates_df.reset_index(inplace=True)
    df = pd.merge(df, aggregates_df, how='left', on=['team_yearID', 'team_teamID'])
    return df


def consolidate_yearly_player_data(df):
    """
    For players who played for multiple teams in a year, consolidate that stats onto a single row.

    :param df: pandas dataframe
    :returns: pandas dataframe
    """
    df['player_year_dupe'] = df.duplicated(subset=['playerID', 'yearID'], keep=False)
    dupe_df = df.loc[df['player_year_dupe'] == True]
    df = df.loc[df['player_year_dupe'] != True]

    min_df = dupe_df.groupby(['playerID', 'yearID']).agg({'order': 'min'})
    min_df.reset_index(inplace=True)

    agg_df = dupe_df.groupby(['playerID', 'yearID']).agg(['sum'])
    agg_df.reset_index(inplace=True)
    agg_df = agg_df.droplevel(1, axis=1)
    agg_df.drop(['order', 'teamID', 'lgID', 'player_year_dupe'], 1, inplace=True)

    dupe_df = pd.merge(dupe_df, min_df, how='inner', on=['playerID', 'yearID', 'order'])
    dupe_df = pd.merge(dupe_df, agg_df, how='inner', on=['playerID', 'yearID'])
    dupe_df = dupe_df[dupe_df.columns.drop(list(dupe_df.filter(regex='_x')))]
    dupe_df.columns = dupe_df.columns.str.rstrip('_y')
    df = pd.concat([df, dupe_df], axis=0)

    df.sort_values(by=['playerID', 'yearID', 'order'], ascending=True, inplace=True)
    df.drop('player_year_dupe', 1, inplace=True)
    return df


def calculate_batting_stats(player_df, batting_df):
    """
    Omnibus function for calculating batting stats.

    :param player_df: dataframe of general player attributes
    :param batting_df: dataframe of batting stats
    :returns: pandas dataframe
    """
    batting_df.reset_index(inplace=True)
    batting_df.rename(columns={'index': 'order'}, inplace=True)
    batting_df = consolidate_yearly_player_data(batting_df)
    batting_df = calculate_expanding_obp(batting_df)
    batting_df.reset_index(inplace=True, drop=True)
    batting_df = find_player_age(player_df, batting_df)
    return batting_df


def calculate_pitching_stats(player_df, pitching_df):
    """
    Omnibus function for calculating batting stats.

    :param player_df: dataframe of general player attributes
    :param pitching_df: dataframe of pitching stats
    :returns: pandas dataframe
    """
    pitching_df.reset_index(inplace=True)
    pitching_df.rename(columns={'index': 'order'}, inplace=True)
    pitching_df = consolidate_yearly_player_data(pitching_df)
    pitching_df = calculate_expanding_era(pitching_df)
    pitching_df.reset_index(inplace=True, drop=True)
    pitching_df = find_player_age(player_df, pitching_df)
    return pitching_df


def append_all_star_appearances(all_star_df, batting_df, pitching_df):
    """
    Appends all-star appearances to batting and and pitching dataframes.

    :param all_star_df: dataframe of all-star appearances
    :param batting_df: dataframe of batting stats
    :param pitching_df: dataframe of pitching stats
    :returns: pandas dataframe
    """
    all_star_df = find_total_and_lagged_all_star_appearances(all_star_df, 5)
    batting_df = pd.merge(batting_df, all_star_df, how='left', on=['playerID', 'yearID'])
    pitching_df = pd.merge(pitching_df, all_star_df, how='left', on=['playerID', 'yearID'])
    return batting_df, pitching_df


def add_column_name_prefixes(team_records_df, batting_df, pitching_df):
    """
    Adds prefixes to every column in a set of dataframes.

    :param team_records_df: dataframe containing yearly team records
    :param batting_df: dataframe of batting stats
    :param pitching_df: dataframe of pitching stats
    :returns: tuple of three pandas dataframes
    """
    team_records_df = team_records_df.add_prefix('team_')
    batting_df = batting_df.add_prefix('batting_')
    pitching_df = pitching_df.add_prefix('pitching_')
    return team_records_df, batting_df, pitching_df


def merge_dataframes(team_records_df, postseason_df, batting_df, pitching_df):
    """
    Merges team records and postseason results onto batting and pitching stats.

    :param team_records_df: dataframe of team records
    :param postseason_df: dataframe of postseason results
    :param batting_df: dataframe of batting stats
    :param pitching_df: dataframe of pitching stats
    :returns: tuple of three dataframes
    """
    teams_df = pd.merge(team_records_df, postseason_df, how='left', left_on=['team_teamID', 'team_yearID'],
                        right_on=['teamID', 'yearID'])
    teams_batting_df = pd.merge(teams_df, batting_df, how='left', left_on=['team_teamID', 'team_yearID'],
                                right_on=['batting_teamID', 'batting_yearID'])
    teams_pitching_df = pd.merge(teams_df, pitching_df, how='left', left_on=['team_teamID', 'team_yearID'],
                                 right_on=['pitching_teamID', 'pitching_yearID'])
    return teams_df, teams_batting_df, teams_pitching_df


def prep_team_level_dataframes(team_records_df, postseason_df):
    """
    Omnibus function to prepare team records and postseason dataframes.

    :param team_records_df: dataframe of team records
    :param postseason_df: dataframe of postseason results
    :returns: tuple of dataframes
    """
    team_records_df = find_yearly_team_winning_percentages(team_records_df, 5)
    postseason_df = find_postseason_results(postseason_df, 5)
    return team_records_df, postseason_df


def create_modeling_dataframe(teams_df, teams_batting_df, teams_pitching_df, target_df):
    """
    Creates dataframe for modeling. This consists of one row for every team for every year. The target is the team who
    won the world series in the upcoming year.

    """
    teams_df = pd.concat([teams_df, teams_batting_df, teams_pitching_df], axis=0)
    teams_df.reset_index(inplace=True, drop=True)
    teams_df = teams_df.loc[teams_df['team_yearID'] >= 1905]
    teams_df = teams_df.loc[~teams_df['team_yearID'].isin([1919, 1994])]
    teams_df = teams_df[[
        'team_yearID', 'team_teamID', 'team_winning_percentage_lag_1', 'team_winning_percentage_lag_2',
        'team_winning_percentage_lag_3', 'team_winning_percentage_lag_4', 'team_winning_percentage_lag_5',
        'team_winning_percentage_rolling_3', 'team_winning_percentage_rolling_5',
        'postseason_result_last_year_lag_1_year', 'postseason_result_last_year_lag_2_year',
        'postseason_result_last_year_lag_3_year', 'postseason_result_last_year_lag_4_year',
        'postseason_result_last_year_lag_5_year', 'batting_ops', 'pitching_era',
        'batting_as_appearance_last_year_lag_1_year', 'batting_as_appearance_last_year_lag_3_year',
        'batting_expanding_as_appearances', 'pitching_as_appearance_last_year_lag_1_year',
        'pitching_as_appearance_last_year_lag_3_year', 'pitching_expanding_as_appearances',
        'batting_player_age', 'pitching_player_age'
        ]]
    teams_df = flatten_data_to_team_and_year(teams_df)
    teams_df = pd.merge(teams_df, target_df, how='left', left_on=['team_yearID', 'team_teamID'],
                        right_on=['yearID', 'teamIDwinner'])
    teams_df.fillna({'target': 0}, inplace=True)
    teams_df['target'] = teams_df['target'].astype(int)
    teams_df.fillna(value=0, inplace=True)
    return teams_df


def create_train_test_split(df, target, test_set_start):
    """
    Creates a train-test split, with every row after test_set_start being assigned to the test set.

    :param df: pandas dataframe
    :param target: name of the target column
    :param test_set_start: year to begin the test set
    """
    test_df = df.loc[df['team_yearID'] >= test_set_start]
    train_df = df.loc[df['team_yearID'] < test_set_start]
    y_train = train_df[target]
    y_test = test_df[target]
    x_train = train_df.drop(target, 1)
    x_test = test_df.drop(target, 1)
    return x_train, x_test, y_train, y_test


def drop_columns(df, drop_cols):
    """
    Drops columns from a pandas dataframe

    :param df: pandas dataframe
    :param drop_cols: list of columns to drop
    :returns: pandas dataframe
    """
    df = df.drop(drop_cols, 1)
    return df


def subtract_columns(df, col1, col2):
    """
    Subtracts two pandas dataframe columns to create a new column.

    :param df: pandas dataframe
    :param col1: string name of first column
    :param col2: string name of second column
    :returns: pandas dataframe
    """
    df[f'{col1}_{col2}_diff'] = df[col1] - df[col2]
    return df


class FeaturesToDict(BaseEstimator, TransformerMixin):
    """
    Transformer to cast features to a dictionary.
    """
    def __int__(self):
        pass

    def fit(self, X, Y=None):
        self.fitted_ = True
        return self

    def transform(self, X, Y=None):
        if isinstance(X, np.ndarray):
            X = pd.DataFrame(X)
        X = X.to_dict(orient='records')
        return X
