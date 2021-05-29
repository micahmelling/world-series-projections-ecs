from sklearn.pipeline import Pipeline
from sklearn.feature_extraction import DictVectorizer
from sklearn.preprocessing import FunctionTransformer

from helpers.model_helpers import FeaturesToDict, drop_columns, subtract_columns
from modeling.config import DROP_COLUMNS


def construct_pipeline(model):
    pipeline = Pipeline(steps=
    [
        ('feature_dropper', FunctionTransformer(drop_columns, validate=False,
                                                kw_args={'drop_cols': DROP_COLUMNS})),
        ('player_age_diff', FunctionTransformer(subtract_columns, validate=False,
                                                kw_args={'col1': 'batting_player_age',
                                                         'col2': 'pitching_player_age'})),
        ('dict_creator', FeaturesToDict()),
        ('dict_vectorizer', DictVectorizer()),
        ('model', model)
    ])
    return pipeline
