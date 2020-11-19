import pandas as pd

from lr_customer_value.config import config
from lr_customer_value.processing.data_management import load_pipeline
from lr_customer_value.processing.validation import validate_inputs
from lr_customer_value import __version__ as _version

import logging

_logger = logging.getLogger(__name__)

pipeline_file_name = f"{config.PIPELINE_SAVE_FILE}_{_version}.pkl"
_redhat_pipe = load_pipeline(file_name=pipeline_file_name)


def make_prediction(*, input_data) -> dict:
    """Makes a prediction using saved model pipeline."""

    _data = pd.DataFrame(input_data)
    validated_data = validate_inputs(input_data=_data)
    prediction = _redhat_pipe.predict(validated_data[config.FEATURES])
    results = {"prediction": prediction, "version": _version}

    _logger.info(
        f"Prediction in process using model version: {_version} "
        f"Inputs (5 samples): {validated_data[0:5]} "
        f"Predictions: {results['prediction'][0:5]}"
    )

    return results
