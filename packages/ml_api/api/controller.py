from flask import Blueprint, request, jsonify, render_template, \
    flash, redirect, url_for, current_app, send_from_directory
from lr_customer_value.predict import make_prediction, make_batch_prediction
from lr_customer_value.processing.data_management import load_dataset
from lr_customer_value import __version__ as _version
from lr_customer_value.config import config as model_config

import os
import pandas as pd
import numpy as np
import json

from api.config import get_logger, UPLOAD_FOLDER
from api.validation import validate_inputs, allowed_file, validate_clients_exist
from api import __version__ as api_version

from werkzeug.utils import secure_filename

_logger = get_logger(logger_name=__name__)

prediction_app = Blueprint('prediction_app', __name__)
user_data_folder = UPLOAD_FOLDER


@prediction_app.route('/', methods=['GET'])
def health():
    if request.method == 'GET':
        return render_template("index.html")


@prediction_app.route('/version', methods=['GET'])
def version():
    if request.method == 'GET':
        return jsonify({'model_version': _version,
                        'api_version': api_version})


@prediction_app.route('/v1/predict/customer_value', methods=['GET', 'POST'])
def predict():
    errors = []
    if request.method == 'POST':
        # Step 1: Extract POST data from request body as JSON
        try:
            json_data = request.form.to_dict()
        except:
            errors.append(
                "Unable to get URL. Please make sure it's valid and try again."
            )
            return render_template('v1/predict/predict.html', errors=errors)

        _logger.debug(f'Inputs: {json_data}')
        # Step 2: Validate the input using marshmallow schema
        input_data, errors = validate_inputs(input_data=json_data)
        # Step 3: Model prediction
        result = make_prediction(input_data=[input_data])
        _logger.debug(f'Outputs:  {result}')

        # Step 4
        # for json to be able to serialize the output,
        # we convert numpy ndarray to list

        prediction = result.get('prediction').tolist()
        version = result.get('version')
        result = {'prediction': str(prediction[0]), 'version': version}
        tmp = [result, json_data]
        print(tmp[0], tmp[1])
        return render_template('v1/predict/predict.html', results=result, errors=errors)

        # Step 5: Return the response as JSON
    else:
        result = {}
        return render_template('v1/predict/predict.html', results=result, errors=errors)


# results=jsonify({'prediction': prediction, 'version': version, 'errors': errors}))


@prediction_app.route('/v1/predict/upload', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        # check if the post request has the file part
        if 'test_file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['test_file']
        # if user does not select file, browser also
        # submit an empty part without filename
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)

        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            print(filename)
            # Step 0
            # clean previous user data
            for root, dirs, files in os.walk(user_data_folder):
                for _ in files:
                    os.remove(os.path.join(user_data_folder, _))
            # save new data
            file.save(os.path.join(user_data_folder, filename))
            # Step 2
            # first check if the clients in this file all exist in our database
            # this is part of api validation code
            # if not, we don't give prediction for these records
            input_data, client_flags = validate_clients_exist(filename)
            output_size = len(client_flags)
            invalid_size = output_size - len(input_data)
            # print(f'{len(input_data)} out of {output_size} are valid')
            # Step 3
            # validate schema using marshmallow
            json_data = input_data.to_json(orient='records')
            _data = json.loads(json_data)

            input_data, errors = validate_inputs(input_data=_data)
            if errors:
                # so I know what type of object it is
                # print('errors', errors)
                render_template('/v1/predict/upload.html', errors=errors)
            # Step 4
            # Model prediction
            results = make_batch_prediction(input_data=input_data)
            # Step 5
            # Save the results in a csv file
            df = pd.DataFrame(data=np.zeros(output_size, ), index=None,
                              dtype=int,
                              columns=['outcome'])
            # if False, the row is valid
            df.loc[np.where(~ client_flags)[0], ['outcome']] = \
                np.reshape(results['prediction'], (len(input_data), 1))
            df.loc[np.where(client_flags)[0], ['outcome']] = \
                np.reshape([np.nan] * invalid_size, (invalid_size, 1))
            # now save it to file
            df.to_csv(os.path.join(user_data_folder, 'predictions.csv'))
            # return redirect(url_for('.uploaded_file', x='preds_' + filename))
            return render_template('/v1/predict/download.html', errors=errors, filename='predictions.csv')
    return render_template('/v1/predict/upload.html')


@prediction_app.route('/download/<x>')
def download(x):
    return send_from_directory(current_app.config['UPLOAD_FOLDER'], x)