import numpy as np
# path
import pathlib

import lr_customer_value

PACKAGE_ROOT = pathlib.Path(lr_customer_value.__file__).resolve().parent
TRAINED_MODEL_DIR = PACKAGE_ROOT / 'trained_models'
DATASET_DIR = PACKAGE_ROOT / 'datasets'

# data
TRAINING_DATA_FILES = [DATASET_DIR / 'act_train.csv', DATASET_DIR / 'people.csv']
TESTING_DATA_FILES = [DATASET_DIR / 'act_test.csv', DATASET_DIR / 'people.csv']


PIPELINE_NAME = 'logistic_reg_redhat'
PIPELINE_SAVE_FILE = 'lr_customer_value'

TARGET = 'outcome'

# input variables (after merging)
FEATURES = ['activity_category', 'char_1_y', 'char_2_y', 'char_3_y', 'char_4_y',
            'char_5_y', 'char_6_y', 'char_7_y', 'char_8_y', 'char_9_y',
            'char_10_y', 'char_12', 'char_13', 'char_16', 'char_20',
            'char_22', 'char_23', 'char_25', 'char_26', 'char_27',
            'char_30', 'char_32', 'char_33', 'char_34', 'char_35',
            'char_36', 'char_38', 'char_5_x',
            # this one is to extract year:
            'date_x']

# this variable is to extract activity year,
# must be dropped afterwards
# DROP_FEATURES = 'date_x'

# numerical variables with NA in train set

# categorical variables with NA in train set
CATEGORICAL_VARS_WITH_NA = ['char_5_x']


# variables to use as time
TEMPORAL_VARS = ['date_x']

# variables to convert to categorical
NUM_TO_CAT_VARS = ['char_38']

# variables with high cardinality
HIGH_DIM_DICT = {'char_38': [0, 0.4, 1]}

# categorical variables to encode
CATEGORICAL_VARS = ['activity_category', 'char_1_y', 'char_2_y', 'char_3_y', 'char_4_y',
                    'char_5_y', 'char_6_y', 'char_7_y', 'char_8_y', 'char_9_y',
                    'char_10_y', 'char_12', 'char_13', 'char_16', 'char_20',
                    'char_22', 'char_23', 'char_25', 'char_26', 'char_27',
                    'char_30', 'char_32', 'char_33', 'char_34', 'char_35',
                    'char_36', 'char_5_x',
                    # this one is to extract year:
                    'date_x']

# only one variable is allowed to have NA values
# could add a numerical imputer to the pipeline based on mode of char_38

CATEGORICAL_NA_NOT_ALLOWED = [
    feature for feature in CATEGORICAL_VARS if feature not in CATEGORICAL_VARS_WITH_NA
]

NUMERICAL_NA_NOT_ALLOWED = [
    feature for feature in FEATURES if feature not in CATEGORICAL_VARS
]

OUTCOME_200_220 = np.array([0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0])
