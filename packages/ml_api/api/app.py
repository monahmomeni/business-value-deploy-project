from flask import Flask

from api.config import get_logger, UPLOAD_FOLDER


_logger = get_logger(logger_name=__name__)


def create_app(*, config_object) -> Flask:
    """Create a flask app instance"""

    flask_app = Flask('ml_api')
    flask_app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
    flask_app.config['MAX_CONTENT_LENGTH'] = 300 * 1024 * 1024

    flask_app.config.from_object(config_object)

    # import blueprints
    from api.controller import prediction_app
    flask_app.register_blueprint(prediction_app)
    _logger.debug('Application instance created')

    return flask_app
