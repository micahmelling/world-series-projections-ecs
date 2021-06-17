import pandas as pd

from ds_helpers import aws, db


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


def get_password_for_username(username, db_secret):
    query = f'''
    select password
    from world_series.app_credentials
    where username = '{username}';
    '''
    # mysql_conn_dict = aws.get_secrets_manager_secret(db_secret)
    mysql_conn_dict = {
        'host': 'churn-databse.cdbxl5rdojax.us-west-2.rds.amazonaws.com',
        'user': 'churn_model',
        'password': 'adfVg6a_hbzwrdaYa_kbZ',
        'database': 'world_series'
    }
    df = pd.read_sql(query, db.connect_to_mysql(mysql_conn_dict))
    return df['password'][0]
