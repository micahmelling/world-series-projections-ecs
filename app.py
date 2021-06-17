import uuid
import joblib

from hashlib import sha256
from copy import deepcopy
from flask import Flask, session, request, redirect, url_for, render_template, flash
from flask_talisman import Talisman
from ds_helpers import aws

from app_settings import APP_SECRET, LOGGING_S3_BUCKET_NAME, AWS_KEYS_SECRET_NAME, ENVIRONMENT, DATABASE_SECRET
from helpers.app_helpers import log_payload_to_s3, produce_prediction_json, get_current_timestamp
from data.data import get_password_for_username


def initialize_app():
    app = Flask(__name__)
    global model
    model = joblib.load('model.pkl')
    global data
    data = joblib.load('data/data.pkl')
    if ENVIRONMENT == 'local':
        aws.set_aws_environment_variables(AWS_KEYS_SECRET_NAME)
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


@app.route("/login", methods=["POST", "GET"])
def login():
    if request.method == 'POST':
        form_submission = request.form
        username = str(form_submission['username'])
        password = str(form_submission['password'])
        hashed_password = sha256(password.encode('utf-8')).hexdigest()
        database_password = get_password_for_username(username, DATABASE_SECRET)
        if hashed_password == database_password:
            session['logged_in'] = True
            return redirect(url_for('model_interface'))
        else:
            flash('Credentials are not valid. Please try again.')
    return render_template('login.html')


@app.route("/logout", methods=["POST", "GET"])
def logout():
    if request.method == 'POST':
        session['logged_in'] = False
        return redirect(url_for('login'))
    return render_template('logout.html')


@app.route('/model_interface', methods=['GET', 'POST'])
def model_interface():
    logged_in = session.get('logged_in', False)
    if logged_in:
        if request.method == 'POST':
            form_submission = request.form
            selected_year = int(form_submission['year'])
            year_data = data.loc[data['team_yearID'] == selected_year]
            if len(year_data) > 0:
                predictions = produce_prediction_json(model, year_data)
                return render_template('model_interface.html', predictions=predictions)
            else:
                return render_template('model_interface.html',
                                       predictions=f'Predictions for {selected_year} could not be produced.')
        else:
            return render_template('model_interface.html', predictions='predictions will be rendered here')
    return redirect(url_for('login'))


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


if __name__ == "__main__":
    app.run(debug=True)
