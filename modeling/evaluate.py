import pandas as pd
import numpy as np
import os

from helpers.model_helpers import make_directories_if_not_exists


def produce_predictions(pipeline, model_name, x_test, y_test, class_cutoff):
    """
    Produces a dataframe consisting of the probability and class predictions from the model on the test set along
    with the y_test. The dataframe is saved locally.

    :param pipeline: scikit-learn modeling pipeline
    :param model_name: name of the model
    :param x_test: x_test dataframe
    :param y_test: y_test series
    :param class_cutoff: the probability cutoff to separate classes
    :returns: pandas dataframe
    """
    df = pd.concat(
        [
            pd.DataFrame(pipeline.predict_proba(x_test), columns=['0_prob', '1_prob']),
            y_test.reset_index(drop=True),
            x_test[['team_yearID', 'teamIDwinner', 'team_teamID']].reset_index(drop=True)
        ],
        axis=1)
    df['predicted_class'] = np.where(df['1_prob'] >= class_cutoff, 1, 0)
    df = df[['predicted_class'] + [col for col in df.columns if col != 'predicted_class']]
    make_directories_if_not_exists([os.path.join(model_name, 'diagnostics', 'predictions')])
    df.to_csv(os.path.join(model_name, 'diagnostics', 'predictions', 'predictions_vs_actuals.csv'), index=False)
    return df


def _evaluate_model(df, target, predictions, scorer, metric_name):
    """
    Applies a scorer function to evaluate predictions against the ground-truth labels.

    :param df: pandas dataframe containing the predictions and the actuals
    :param target: name of the target column in df
    :param predictions: name of the column with the predictions
    :param scorer: scoring function to evaluate the predictions
    :param metric_name: name of the metric we are using to score our model

    :returns: pandas dataframe
    """
    score = scorer(df[target], df[predictions])
    df = pd.DataFrame({metric_name: [score]})
    return df


def run_evaluation_metrics(df, target, model_name, evaluation_list):
    """
    Runs a series of evaluations metrics on a model's predictions and writes the results locally.

    :param df: pandas dataframe containing the predictions and the actuals
    :param target: name of the target column
    :param model_name: name of the model
    :param evaluation_list: list of tuples, which each tuple having the ordering of: the column with the predictions,
    the scoring function callable, and the name of the metric
    """
    main_df = pd.DataFrame()
    for metric_config in evaluation_list:
        temp_df = _evaluate_model(df, target, metric_config[0], metric_config[1], metric_config[2])
        main_df = pd.concat([main_df, temp_df], axis=1)
    main_df = main_df.T
    main_df.reset_index(inplace=True)
    main_df.columns = ['scoring_metric', 'holdout_score']
    main_df['model_uid'] = model_name
    main_df['holdout_type'] = 'test'
    make_directories_if_not_exists([os.path.join(model_name, 'diagnostics', 'evaluation')])
    main_df.to_csv(os.path.join(model_name, 'diagnostics', 'evaluation', 'evaluation_scores.csv'), index=False)


def run_omnibus_model_evaluation(pipeline, model_name, x_test, y_test, class_cutoff, target, evaluation_list):
    """
    Runs a series of functions to evaluate a model's performance.

    :param pipeline: scikit-learn pipeline
    :param model_name: name of the model
    :param x_test: x_test dataframe
    :param y_test: y_test series
    :param class_cutoff: the probability cutoff to separate classes
    :param target: name of the target column
    :param evaluation_list: list of tuples, which each tuple having the ordering of: the column with the predictions,
    the scoring function callable, and the name of the metric
    """
    print(f'evaluating {model_name}...')
    predictions_df = produce_predictions(pipeline, model_name, x_test, y_test, class_cutoff)
    run_evaluation_metrics(predictions_df, target, model_name, evaluation_list)
