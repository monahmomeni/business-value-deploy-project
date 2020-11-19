import json

from lr_customer_value.config import config
from lr_customer_value.processing.data_management import load_dataset


def test_prediction_endpoint_validation_200(flask_test_client):
    # Given
    # Load the test data from the lr_customer_value package.
    # This is important as it makes it harder for the test
    # data versions to get confused by not spreading it
    # across packages.
    test_data = load_dataset(files_list=config.TESTING_DATA_FILES)
    if len(test_data) > 10000:
        test_data = test_data[:10000]
        
    post_json = test_data.to_json(orient='records')

    # When
    response = flask_test_client.post('/v1/predict/customer_value',
                                      json=json.loads(post_json))

    # Then
    assert response.status_code == 200
    response_json = json.loads(response.data)

    # Check correct number of errors removed
    if response_json.get('errors'):
        assert len(response_json.get('prediction')) + len(
            response_json.get('errors')) == len(test_data)
    else:
        assert len(response_json.get('prediction')) == len(test_data)

