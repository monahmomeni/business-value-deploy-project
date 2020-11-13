import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
import joblib

from lr_customer_value import pipeline
from lr_customer_value.config import config
from lr_customer_value.processing.data_management import load_dataset, save_pipeline
from lr_customer_value import __version__ as _version

import logging

_logger = logging.getLogger(__name__)


def run_training() -> None:
    """Train the model"""

    # read training data
    data = load_dataset(files_list=config.TRAINING_DATA_FILES)

    # divide train and test
    x_train, x_test, y_train, y_test = train_test_split(
        data[config.FEATURES],
        data[config.TARGET],
        test_size=0.1,
        random_state=0)

    pipeline.redhat_pipe.fit(x_train[config.FEATURES], y_train)

    _logger.info(f"saving model version: {_version}")
    save_pipeline(pipeline_to_persist=pipeline.redhat_pipe)


if __name__ == '__main__':
    run_training()
