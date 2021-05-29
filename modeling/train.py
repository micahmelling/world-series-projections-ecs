import warnings
import time

from sklearn.metrics import log_loss

from data.data import get_postseason_results, get_batting_stats, get_pitching_stats, \
    get_historical_all_star_appearances, get_player_info, get_team_records, get_fielding_positions
from helpers.model_helpers import create_target_dataframe, clean_batting_and_pitching_players, calculate_batting_stats, \
    calculate_pitching_stats, append_all_star_appearances, add_column_name_prefixes, merge_dataframes, \
    create_modeling_dataframe, create_train_test_split, make_directories_if_not_exists, create_uid, \
    prep_team_level_dataframes
from modeling.config import TARGET, TEST_SET_START_YEAR, MODEL_TRAINING_LIST, CV_SCORING, CLASS_CUTOFF, \
    MODEL_EVALUATION_LIST, CV_SPLITS
from modeling.model import train_model
from modeling.pipeline import construct_pipeline
from modeling.evaluate import run_omnibus_model_evaluation
from modeling.explain import run_omnibus_model_explanation


warnings.filterwarnings('ignore')


def assemble_modeling_data():
    """
    Assembles training data for predicting the World Series winner based on statistics only from previous seasons.
    """
    batting_df = get_batting_stats()
    pitching_df = get_pitching_stats()
    all_star_df = get_historical_all_star_appearances()
    postseason_df = get_postseason_results()
    team_records_df = get_team_records()
    player_df = get_player_info()
    positions_df = get_fielding_positions()
    target_df = create_target_dataframe(postseason_df)
    batting_df, pitching_df = clean_batting_and_pitching_players(batting_df, pitching_df, positions_df)
    team_records_df, postseason_df = prep_team_level_dataframes(team_records_df, postseason_df)
    batting_df = calculate_batting_stats(player_df, batting_df)
    pitching_df = calculate_pitching_stats(player_df, pitching_df)
    batting_df, pitching_df = append_all_star_appearances(all_star_df, batting_df, pitching_df)
    team_records_df, batting_df, pitching_df = add_column_name_prefixes(team_records_df, batting_df, pitching_df)
    teams_df, teams_batting_df, teams_pitching_df = merge_dataframes(team_records_df, postseason_df, batting_df,
                                                                     pitching_df)
    teams_df = create_modeling_dataframe(teams_df, teams_batting_df, teams_pitching_df, target_df)
    return teams_df


def train_and_evaluate_models(x_train, x_test, y_train, y_test):
    """
    Trains and evaluates a series of machine learning models.
    """
    for model in MODEL_TRAINING_LIST:
        model_uid = create_uid(model.model_name)
        make_directories_if_not_exists([model_uid])
        pipeline = train_model(x_train, y_train, construct_pipeline, model_uid, model.model, model.param_grid,
                               model.iterations, CV_SPLITS, CV_SCORING)
        run_omnibus_model_evaluation(pipeline, model_uid, x_test, y_test, CLASS_CUTOFF, TARGET, MODEL_EVALUATION_LIST)
        run_omnibus_model_explanation(pipeline, x_test, y_test, x_train, y_train, log_loss, CV_SCORING, 'probability',
                                      model_uid, True)


def main():
    """
    Main execution function.
    """
    modeling_df = assemble_modeling_data()
    x_train, x_test, y_train, y_test = create_train_test_split(modeling_df, TARGET, TEST_SET_START_YEAR)
    train_and_evaluate_models(x_train, x_test, y_train, y_test)


if __name__ == "__main__":
    start_time = time.time()
    print('running model training script...')
    main()
    print('--- the script took {} minutes to complete --'.format(str((time.time() - start_time) / 60)))
