import pandas as pd


def get_postseason_results():
    return pd.read_csv('https://github.com/chadwickbureau/baseballdatabank/blob/master/core/SeriesPost.csv?raw=true')


def get_batting_stats():
    return pd.read_csv('https://github.com/chadwickbureau/baseballdatabank/blob/master/core/Batting.csv?raw=true')


def get_pitching_stats():
    return pd.read_csv('https://github.com/chadwickbureau/baseballdatabank/blob/master/core/Pitching.csv?raw=true')


def get_fielding_positions():
    return pd.read_csv('https://github.com/chadwickbureau/baseballdatabank/blob/master/core/Appearances.csv?raw=true')


def get_historical_all_star_appearances():
    return pd.read_csv('https://github.com/chadwickbureau/baseballdatabank/blob/master/core/AllstarFull.csv?raw=true')


def get_player_info():
    return pd.read_csv('https://github.com/chadwickbureau/baseballdatabank/blob/master/core/People.csv?raw=true')


def get_team_records():
    return pd.read_csv('https://github.com/chadwickbureau/baseballdatabank/blob/master/core/Teams.csv?raw=true')
