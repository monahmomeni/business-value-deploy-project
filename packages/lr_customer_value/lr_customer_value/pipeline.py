from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler

from lr_customer_value.processing import preprocessors as pp
from lr_customer_value.config import config
from lr_customer_value.processing import features

import logging

_logger = logging.getLogger(__name__)

redhat_pipe = Pipeline(
    [
        ('categorical_imputer',
         pp.CategoricalImputer(variables=config.CATEGORICAL_VARS_WITH_NA)),

        ('temporal_variable',
         features.TemporalVariableEstimator(variables=config.TEMPORAL_VARS)),

        ('convert_to_object',
         pp.ChangeDataType(
            target_type=object,
            variables=config.CATEGORICAL_VARS + list(config.HIGH_DIM_DICT.keys()))),

        ('rare_label_encoder',
         pp.RareLabelCategoricalEncoder(
             tol=0.01,
             variables=config.CATEGORICAL_VARS)),

        ('high_dim_encoder',
         features.MergeHighDimCategories(variables=config.HIGH_DIM_DICT)),

        ('categorical_encoder',
         pp.CategoricalEncoder(variables=config.CATEGORICAL_VARS + list(config.HIGH_DIM_DICT.keys()))),

        ('scaler', MinMaxScaler()),

        ('linear_model', LogisticRegression(C=0.005,
                                            penalty='l2',
                                            dual=False,
                                            max_iter=2000,
                                            random_state=0))
    ]
)
