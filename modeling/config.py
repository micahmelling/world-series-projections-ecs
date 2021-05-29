from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, ExtraTreesClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from hyperopt import hp
from sklearn.metrics import log_loss, brier_score_loss, roc_auc_score, f1_score, balanced_accuracy_score
from collections import namedtuple


TARGET = 'target'
TEST_SET_START_YEAR = 2017
CV_SCORING = 'neg_log_loss'
CLASS_CUTOFF = 0.5
CV_SPLITS = 3
DROP_COLUMNS = ['team_yearID', 'team_teamID', 'yearID', 'teamIDwinner']


FOREST_PARAM_GRID = {
    'model__max_depth': hp.uniformint('model__max_depth', 3, 16),
    'model__min_samples_leaf': hp.uniform('model__min_samples_leaf', 0.001, 0.01),
    'model__max_features': hp.choice('model__max_features', ['log2', 'sqrt'])
}


GRADIENT_BOOSTING_PARAM_GRID = {
    'model__learning_rate': hp.uniform('model__learning_rate', 0.01, 0.5),
    'model__n_estimators': hp.uniformint('model__n_estimators', 75, 150),
    'model__max_depth': hp.uniformint('model__max_depth', 2, 16)
}


XGBOOST_PARAM_GRID = {
    'model__learning_rate': hp.uniform('model__learning_rate', 0.01, 0.5),
    'model__n_estimators': hp.randint('model__n_estimators', 75, 150),
    'model__max_depth': hp.randint('model__max_depth', 2, 16),
    'model__min_child_weight': hp.uniformint('model__min_child_weight', 2, 16)
}


LIGHTGBM_PARAM_GRID = {
    'model__learning_rate': hp.uniform('model__learning_rate', 0.01, 0.5),
    'model__n_estimators': hp.randint('model__n_estimators', 75, 150),
    'model__max_depth': hp.randint('model__max_depth', 2, 16),
    'model__num_leaves': hp.randint('model__num_leaves', 10, 100)
}


model_named_tuple = namedtuple('model_config', {'model_name', 'model', 'param_grid', 'iterations'})
MODEL_TRAINING_LIST = [
    model_named_tuple(model_name='random_forest', model=RandomForestClassifier(), param_grid=FOREST_PARAM_GRID,
                      iterations=100),
    model_named_tuple(model_name='extra_trees', model=ExtraTreesClassifier(), param_grid=FOREST_PARAM_GRID,
                      iterations=100),
    model_named_tuple(model_name='gradient_boosting', model=GradientBoostingClassifier(),
                      param_grid=GRADIENT_BOOSTING_PARAM_GRID, iterations=100),
    model_named_tuple(model_name='xgboost', model=XGBClassifier(), param_grid=XGBOOST_PARAM_GRID, iterations=100),
    model_named_tuple(model_name='light_gbm', model=LGBMClassifier(), param_grid=LIGHTGBM_PARAM_GRID, iterations=100),
]


MODEL_EVALUATION_LIST = [
    ('1_prob', log_loss, 'log_loss'),
    ('1_prob', brier_score_loss, 'brier_score'),
    ('1_prob', roc_auc_score, 'roc_auc'),
    ('predicted_class', f1_score, 'f1'),
    ('predicted_class', balanced_accuracy_score, 'balanced_accuracy'),
]
