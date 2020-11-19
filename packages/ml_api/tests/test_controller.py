from lr_customer_value.config import config as model_config
from lr_customer_value.processing.data_management import load_dataset
from lr_customer_value import __version__ as _version
from lr_customer_value.config import config

import json
import numpy as np

from api import __version__ as api_version


def test_health_endpoint_returns_200(flask_test_client):
    # When
    response = flask_test_client.get('/health')

    # Then
    assert response.status_code == 200


def test_version_endpoint_returns_version(flask_test_client):
    # When
    response = flask_test_client.get('/version')

    # Then
    assert response.status_code == 200
    response_json = json.loads(response.data)
    assert response_json['model_version'] == _version
    assert response_json['api_version'] == api_version


def test_prediction_endpoint_returns_prediction(flask_test_client):

    # Given
    # Load the test data from the lr_customer_model package
    # This is important as it makes it harder for the test
    # data versions to get confused by not spreading it
    # across packages.

    test_data = load_dataset(files_list=model_config.TESTING_DATA_FILES)
    post_json = test_data[200:220].to_json(orient='records')

    # When
    response = flask_test_client.post('/v1/predict/customer_value',
                                      json=json.loads(post_json))

    # Then
    assert response.status_code == 200
    response_json = json.loads(response.data)
    prediction = response_json['prediction']
    response_version = response_json['version']
    assert np.array_equal(prediction, config.OUTCOME_200_220)
    assert response_version == _version

