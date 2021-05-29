import shap
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import multiprocessing as mp

from statistics import mean
from tqdm import tqdm
from sklearn.inspection import permutation_importance
from copy import deepcopy
from functools import partial
from sklearn.inspection import plot_partial_dependence
from PyALE import ale

from helpers.model_helpers import make_directories_if_not_exists


plt.switch_backend('Agg')


def _run_shap_explainer(x_df, explainer, boosting_model):
    """
    Runs the SHAP explainer on a dataframe.

    :param x_df: x dataframe
    :param explainer: SHAP explainer object
    :param boosting_model: Boolean of whether or not the explainer is for a boosting model
    """
    if boosting_model:
        return explainer.shap_values(x_df)
    else:
        return explainer.shap_values(x_df)[1]


def _run_parallel_shap_explainer(x_df, explainer, boosting_model):
    """
    Splits x_df into evenly-split partitions based on the number of CPU available on the machine. Then, the SHAP
    explainer object is run in parallel on each subset of x_df. The results are then combined into a single object.

    :param x_df: x dataframe
    :param explainer: SHAP explainer object
    :param boosting_model: Boolean of whether or not the explainer is for a boosting model
    """
    array_split = np.array_split(x_df, mp.cpu_count())
    shap_fn = partial(_run_shap_explainer, explainer=explainer, boosting_model=boosting_model)
    with mp.Pool(processes=mp.cpu_count()) as pool:
        result = pool.map(shap_fn, array_split)
    result = np.concatenate(result)
    return result


def _produce_raw_shap_values(model, model_uid, x_df, calibrated, boosting_model):
    """
    Produces the raw shap values for every observation in the test set. A dataframe of the shap values is saved locally
    as a csv. The shap expected value is extracted and save locally in a csv.

    :param model: fitted model
    :param model_uid: model uid
    :param x_df: x dataframe
    :param calibrated: boolean of whether or not the model is a CalibratedClassifierCV; the default is False
    :param boosting_model: Boolean of whether or not the explainer is for a boosting model
    :returns: numpy array
    """
    if calibrated:
        print('generating SHAP values for a calibrated classifier...this might take a while...')
        shap_values_list = []
        shap_expected_list = []
        for calibrated_classifier in tqdm(model.calibrated_classifiers_):
            explainer = shap.TreeExplainer(calibrated_classifier.base_estimator)
            shap_values = _run_parallel_shap_explainer(x_df, explainer, boosting_model)
            shap_expected_value = explainer.expected_value
            shap_values_list.append(shap_values)
            shap_expected_list.append(shap_expected_value)
        shap_values = np.array(shap_values_list).sum(axis=0) / len(shap_values_list)
        shap_expected_value = mean(shap_expected_list)
        shap_df = pd.DataFrame(shap_values, columns=list(x_df))
        shap_df.to_csv(os.path.join(model_uid, 'diagnostics', 'shap', 'shap_values.csv'), index=False)
        shap_expected_value = pd.DataFrame({'expected_value': [shap_expected_value]})
        shap_expected_value.to_csv(os.path.join(model_uid, 'diagnostics', 'shap', 'shap_expected.csv'), index=False)
    else:
        explainer = shap.TreeExplainer(model)
        shap_values = _run_parallel_shap_explainer(x_df, explainer, boosting_model)
        shap_df = pd.DataFrame(shap_values, columns=list(x_df))
        shap_df.to_csv(os.path.join(model_uid, 'diagnostics', 'shap', 'shap_values.csv'), index=False)
        if boosting_model:
            shap_expected_value = pd.DataFrame({'expected_value': [explainer.expected_value[0]]})
        else:
            try:
                shap_expected_value = pd.DataFrame({'expected_value': [explainer.expected_value[1]]})
            except IndexError:
                shap_expected_value = pd.DataFrame({'expected_value': [explainer.expected_value[0]]})
        shap_expected_value.to_csv(os.path.join(model_uid, 'diagnostics', 'shap', 'shap_expected.csv'), index=False)
    return shap_values


def _generate_shap_global_values(shap_values, x_df, model_uid):
    """
    Extracts the global shape values for every feature ans saves the outcome as a dataframe locally. Amends the
    dataframe so that it could be used in log_feature_importance_to_mysql().

    :param shap_values: numpy array of shap values
    :param x_df: x_df dataframe
    :param model_uid: model uid
    :returns: pandas dataframe
    """
    shap_values = np.abs(shap_values).mean(0)
    df = pd.DataFrame(list(zip(x_df.columns, shap_values)), columns=['feature', 'shap_value'])
    df.sort_values(by=['shap_value'], ascending=False, inplace=True)
    df.to_csv(os.path.join(model_uid, 'diagnostics', 'shap', 'shap_global.csv'), index=False)


def _generate_shap_plot(shap_values, x_df, model_uid, plot_type):
    """
    Generates a plot of shap values and saves it locally.

    :param shap_values: numpy array of shap values produced for x_df
    :param x_df: x dataframe
    :param model_uid: model uid
    :param plot_type: the type of plot we want to generate; generally, either dot or bar
    """
    shap.summary_plot(shap_values, x_df, plot_type=plot_type, show=False)
    plt.savefig(os.path.join(model_uid, 'diagnostics', 'shap', f'shap_values_{plot_type}.png'),
                bbox_inches='tight')
    plt.clf()


def produce_shap_values_and_plots(model, x_df, model_uid, boosting_model, calibrated=False):
    """
    Produces SHAP values for x_df and writes associated diagnostics locally.

    :param model: model with predict method
    :param x_df: x dataframe
    :param model_uid: model uid
    :param boosting_model: Boolean of whether or not the explainer is for a boosting model
    :param calibrated: boolean of whether or not the model is a CalibratedClassifierCV; the default is False
    """
    make_directories_if_not_exists([os.path.join(model_uid, 'diagnostics', 'shap')])
    shap_values = _produce_raw_shap_values(model, model_uid, x_df, calibrated, boosting_model)
    _generate_shap_global_values(shap_values, x_df, model_uid)
    _generate_shap_plot(shap_values, x_df, model_uid, 'dot')
    _generate_shap_plot(shap_values, x_df, model_uid, 'bar')


def run_permutation_importance(pipeline, x_df, y_test, model_uid, scoring, feature_mapping_dict):
    """
    Produces feature permutation importance scores and saved the results locally.

    :param pipeline: fitted pipeline
    :param x_df: x dataframe
    :param y_test: y_test
    :param model_uid: model uid
    """
    result = permutation_importance(pipeline, x_df, y_test, n_repeats=10, random_state=0, scoring=scoring, n_jobs=-1)
    df = pd.DataFrame({
        'permutation_importance_mean': result.importances_mean,
        'permutation_importance_std': result.importances_std,
        'feature': list(x_df)
    })
    df['feature'] = df['feature'].map(feature_mapping_dict)
    df.sort_values(by=['permutation_importance_mean'], ascending=False, inplace=True)
    make_directories_if_not_exists([os.path.join(model_uid, 'diagnostics', 'permutation_importance')])
    df.to_csv(os.path.join(model_uid, 'diagnostics', 'permutation_importance', 'permutation_importance_scores.csv'), 
              index=False)


def _score_drop_col_model(pipe, x_df, y_test, scoring_type, scorer):
    """
    Scores a trained for drop-column feature importance.

    :param pipe: scikit-learn pipeline
    :param x_df: x dataframe
    :param y_test: y_test
    :param scoring_type: if we want to evaluate class or probability predictions
    :param scorer: scikit-learn scoring callable
    :returns: model's score on the test set
    """
    if scoring_type == 'class':
        predictions = pipe.predict(x_df)
        score = scorer(y_test, predictions)
    elif scoring_type == 'probability':
        predictions = pipe.predict_proba(x_df)
        score = scorer(y_test, predictions[:, 1])
    else:
        raise Exception('scoring_type must either be class or probability')
    return score


def _train_and_score_drop_col_model(feature, pipeline, x_train, y_train, x_df, y_test, baseline_score, scoring_type,
                                    scorer):
    """
    Drops specified feature, refits the pipeline to the training data, and determines the differences from the baseline
    model score.

    :param feature: name of the feature to drop
    :param pipeline: fitted scikit-learn pipeline
    :param x_train: x_train
    :param y_train: y_train
    :param x_df: x_df
    :param y_test: y_test
    :param baseline_score: the score on the test set using all the columns for training
    :param scoring_type: if we want to evaluation class or probability predictions
    :param scorer: scikit-learn scoring callable
    """
    try:
        x = x_train.drop(feature, axis=1)
        x_df = x_df.drop(feature, axis=1)
        train_pipe = deepcopy(pipeline)
        train_pipe.fit(x, y_train)
        feature_score = baseline_score - _score_drop_col_model(train_pipe, x_df, y_test, scoring_type, scorer)
    except:
        feature_score = np.nan
    return {'feature': feature, 'importance': feature_score}


def run_drop_column_importance(pipeline, x_train, y_train, x_df, y_test, scorer, scoring_type, model_uid,
                               higher_is_better=True):
    """
    Produces drop column feature importance scores and saves the results locally.

    :param pipeline: fitted pipeline
    :param x_train: x_train
    :param y_train: y_train
    :param x_df: x_df
    :param y_test: y_test
    :param scorer: scoring function
    :param scoring_type: either class or probability
    :param model_uid: model uid
    :param higher_is_better: whether or not a higher score is better
    """
    pipeline_ = deepcopy(pipeline)
    pipeline_.fit(x_train, y_train)
    baseline_score = _score_drop_col_model(pipeline_, x_df, y_test, scoring_type, scorer)
    drop_col_train_fn = partial(_train_and_score_drop_col_model, pipeline=pipeline_, x_train=x_train, y_train=y_train,
                                x_df=x_df, y_test=y_test, baseline_score=baseline_score, scoring_type=scoring_type,
                                scorer=scorer)
    columns = list(x_train)
    with mp.Pool(processes=mp.cpu_count()) as pool:
        result = pool.map(drop_col_train_fn, columns)
    df = pd.DataFrame.from_records(result)
    df.sort_values(by=['importance'], ascending=higher_is_better, inplace=True)
    make_directories_if_not_exists([os.path.join(model_uid, 'diagnostics', 'drop_col_importance')])
    df.to_csv(os.path.join(model_uid, 'diagnostics', 'drop_col_importance', 'drop_column_importance_scores.csv'), 
              index=False)


def _plot_partial_dependence(feature_mapping, model, x_df, plot_kind, model_uid):
    """
    Produces a PDD plot and saves it locally.

    :param feature_mapping: tuple that contains the feature name as the first item and the feature index as the second
    item
    :param model: fitted model
    :param x_df: x dataframe
    :param plot_kind: "both" for ICE plot of "average" for PDP
    :param model_uid: model uid
    """
    _, ax = plt.subplots(ncols=1, figsize=(9, 4))
    display = plot_partial_dependence(model, x_df, [f'f{feature_mapping[1]}'], kind=plot_kind)
    plt.title(feature_mapping[0])
    plt.xlabel(feature_mapping[0])
    plt.savefig(os.path.join(model_uid, 'diagnostics', 'pdp', f'{feature_mapping[0]}_{plot_kind}.png'))
    plt.clf()


def produce_partial_dependence_plots(model, x_df, plot_kind, feature_vocabulary, model_uid):
    """
    Produces a PDP or ICE plot for every column in x_df. x_df is spread across all available CPUs on the machine,
    allowing plots to be created and saved in parallel.

    :param model: fitted model
    :param x_df: x dataframe
    :param plot_kind: "both" for ICE plot of "average" for PDP
    :param feature_vocabulary: list of tuples, with each tuple containing the feature name as the first item and the
    feature index as the second item
    :param model_uid: model uid
    """
    model.fitted_ = True
    make_directories_if_not_exists([os.path.join(model_uid, 'diagnostics', 'pdp')])
    pdp_plot_fn = partial(_plot_partial_dependence, model=model, x_df=x_df, plot_kind=plot_kind, model_uid=model_uid)
    with mp.Pool(processes=mp.cpu_count()) as pool:
        result = pool.map(pdp_plot_fn, feature_vocabulary)


def create_feature_name_mapping_iterable(pipeline, return_tuple=True):
    """
    Creates a feature mapping of the original feature names and the feature names used in the DictVectorizer.

    :param pipeline: scikit-learn pipeline with a dict_vectorizer steps
    :param return_tuple: Boolean of whether to return the dictionary mapping of features or a list of tuples that
    contain the mappings
    :returns: list of tuples or dictionary
    """
    dict_vect = pipeline.named_steps['dict_vectorizer']
    feature_vocabulary = dict_vect.vocabulary_
    if return_tuple:
        vocab_iterable = [(k, v) for k, v in feature_vocabulary.items()]
        return vocab_iterable
    else:
        for key, value in feature_vocabulary.items():
            feature_vocabulary[key] = 'f' + str(value)
        feature_vocabulary = {v: k for k, v in feature_vocabulary.items()}
        return feature_vocabulary


def transform_data_with_pipeline(pipeline, x_df, original_column_names=True):
    """
    Strips out the model from a pipeline and applies the preprocessing steps to x_df.

    :param pipeline: scikit-learn pipeline
    :param x_df: x dataframe
    :param original_column_names: Boolean of whether to use the original column names or the column names used by the
    DictVectorizer
    :returns: pandas dataframe
    """
    pipeline_ = deepcopy(pipeline)
    pipeline_.steps.pop(len(pipeline_) - 1)
    x_df = pipeline_.transform(x_df)
    x_df = pd.DataFrame.sparse.from_spmatrix(x_df)
    x_df = x_df.sparse.to_dense()
    if original_column_names:
        dict_vect = pipeline.named_steps['dict_vectorizer']
        feature_names = list(dict_vect.feature_names_)
        x_df.columns = feature_names
    else:
        x_df = x_df.add_prefix('f')
    return x_df


def _produce_ale_plot(feature_mapping, x_df, model, model_uid):
    """
    Produces an ALE plot and saves it locally.

    :param feature_mapping: tuple containing the feature name as the first item and the feature index as the second item
    :param x_df: x dataframe
    :param model: fitted model
    :param model_uid: model uid
    """
    try:
        ale_effect = ale(X=x_df, model=model, feature=[f'f{feature_mapping[1]}'], include_CI=False)
        plt.title(feature_mapping[0])
        plt.xlabel(feature_mapping[0])
        plt.savefig(os.path.join(model_uid, 'diagnostics', 'ale', f'{feature_mapping[0]}_ale.png'))
        plt.clf()
    except:
        print(f'could not produce ale plot for {feature_mapping[0]}')


def produce_accumulated_local_effects_plots(x_df, model, feature_vocabulary, model_uid):
    """
    Produces an ALE plot for every column numereic column in x_df. x_df is spread across all available CPUs on the
    machine, allowing plots to be created and saved in parallel.

    :param x_df: x dataframe
    :param model: fitted model
    :param feature_vocabulary: list of tuples, with each tuple containing the feature name as the first item and the
    feature index as the second item
    :param model_uid: model uid
    """
    make_directories_if_not_exists([os.path.join(model_uid, 'diagnostics', 'ale')])
    ale_plot_fn = partial(_produce_ale_plot, model=model, x_df=x_df, model_uid=model_uid)
    with mp.Pool(processes=mp.cpu_count()) as pool:
        result = pool.map(ale_plot_fn, feature_vocabulary)


def run_omnibus_model_explanation(pipeline, x_df, y_test, x_train, y_train, scorer, scorer_string, scoring_type,
                                  model_uid, higher_is_better):
    """
    Runs a series of model explainability techniques on the model.
    - PDP plots
    - ICE plots
    - ALE plots
    - permutation importance
    - drop-column importance
    - SHAP values
    
    :param pipeline: scikit-learn pipeline
    :param x_df: x_df
    :param y_test: y_test
    :param x_train: x_train
    :param y_train: y_train
    :param scorer: scikit-learn scoring function
    :param scorer_string: scoring metric in the form of a string (e.g. 'neg_log_loss')
    :param scoring_type: either class or probability
    :param model_uid: model uid
    :param higher_is_better: Boolean of whether or not a higher score is better (e.g. roc auc vs. log loss)
    """
    print(f'explaining {model_uid}...')

    model = pipeline.named_steps['model']
    x_df_original_names_df = transform_data_with_pipeline(pipeline, x_df, True)
    x_df_transformed_names_df = transform_data_with_pipeline(pipeline, x_df, False)
    feature_vocab_iterable = create_feature_name_mapping_iterable(pipeline, True)
    feature_vocab_dict = create_feature_name_mapping_iterable(pipeline, False)

    produce_partial_dependence_plots(model, x_df_transformed_names_df, 'average', feature_vocab_iterable, model_uid)
    produce_partial_dependence_plots(model, x_df_transformed_names_df, 'both', feature_vocab_iterable, model_uid)
    produce_accumulated_local_effects_plots(x_df_transformed_names_df, model, feature_vocab_iterable, model_uid)
    run_permutation_importance(model, x_df_transformed_names_df, y_test, model_uid, scorer_string, feature_vocab_dict)

    model_type = str((type(model))).lower()
    if 'boost' in model_type:
        boosting_model = True
    else:
        boosting_model = False
        
    produce_shap_values_and_plots(model, x_df_original_names_df, model_uid, boosting_model, False)
    run_drop_column_importance(pipeline, x_train, y_train, x_df, y_test, scorer, scoring_type, model_uid,
                               higher_is_better)
