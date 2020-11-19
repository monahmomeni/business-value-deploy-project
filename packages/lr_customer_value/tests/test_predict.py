import numpy as np

from lr_customer_value.predict import make_prediction
from lr_customer_value.processing.data_management import load_dataset
from lr_customer_value.config import config

import logging
_logger = logging.getLogger(__name__)


def test_make_multiple_prediction():
    """
    This function compares the prediction on a records of unseen data
    and compares the results to the ones from research environment experiments
    """

    # Given
    test_data = load_dataset(files_list=config.TESTING_DATA_FILES)
    multiple_json_records = test_data[200:220]

    _logger.info(f"prediction on rows 200-220 of test data.")

    # When
    outcome = make_prediction(input_data=multiple_json_records)

    # Then
    assert outcome is not None
    assert np.array_equal(outcome.get('prediction'), config.OUTCOME_200_220)


def test_data_sanity_check():
    """Checks the validity of data to be consumed by the model"""
    # Given
    test_data = load_dataset(files_list=config.TESTING_DATA_FILES)
    original_data_length = len(test_data)
    multiple_test_json = test_data

    # When
    outcome = make_prediction(input_data=multiple_test_json)

    # Then
    assert outcome is not None
    assert len(outcome.get('prediction')) == 498687

    # We expect some data is filtered out -
    # this logic does not seem very generalizable
    # assert len(outcome.get('prediction')) != original_data_length
