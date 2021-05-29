import uuid
import joblib
import os

from copy import deepcopy
from flask import Flask, session, request
from flask_talisman import Talisman
from ds_helpers import aws

from app_settings import APP_SECRET, LOGGING_S3_BUCKET_NAME, AWS_KEYS_SECRET_NAME
from helpers.app_helpers import log_payload_to_s3, produce_prediction_json, get_current_timestamp


def initialize_app():
    app = Flask(__name__)
    global model
    model = joblib.load('model.pkl')
    global data
    data = joblib.load('data/data.pkl')
    if os.environ['ENVIRONMENT'] == 'local':
        aws.set_aws_environment_variables(AWS_KEYS_SECRET)
    return app


app = initialize_app()
Talisman(app)
app.secret_key = APP_SECRET


@app.before_request
def set_session_uid():
    uid = str(uuid.uuid4())
    session["uid"] = uid


@app.route("/", methods=["POST", "GET"])
def home():
    return "app is healthy"


@app.route("/health", methods=["POST", "GET"])
def health():
    return "app is healthy"


@app.route("/predict", methods=["POST"])
def predict():
    try:
        input_data = request.json
        input_data["uid"] = session.get("uid")
        input_data["url"] = request.url
        input_data["endpoint"] = "predict"
        selected_year = input_data["year"]
        year_data = data.loc[data['team_yearID'] == selected_year]
        predictions = produce_prediction_json(model, year_data)
        output = dict()
        output["prediction"] = predictions
        session["output"] = deepcopy(output)
        session["input"] = input_data
        print(output)
        return output
    except Exception as e:
        print(e)
        output = {
            "error": "app was not able to process request",
            "prediction": 0
        }
        return output
    finally:
        uid = session.get("uid")
        input_payload = session.get("input")
        output_payload = session.get("output")
        output_payload["logging_timestamp"] = str(get_current_timestamp())
        log_payload_to_s3(input_payload, output_payload, uid, LOGGING_S3_BUCKET_NAME)
