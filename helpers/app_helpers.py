import os
import datetime
import pandas as pd
import numpy as np

from ds_helpers import aws


def log_payload_to_s3(input_payload, output_payload, uid, bucket_name):
    new_input_payload = dict()
    new_input_payload['input'] = input_payload
    new_output_payload = dict()
    new_output_payload['output'] = output_payload
    final_payload = str({**new_input_payload, **new_output_payload})
    with open(f'{uid}.json', 'w') as outfile:
        outfile.write(final_payload)
    aws.upload_file_to_s3(f'{uid}.json', bucket_name)
    os.remove(f'{uid}.json')


def produce_prediction_json(model, data):
    predictions_df = pd.concat(
        [
            pd.DataFrame(model.predict_proba(data), columns=['_', 'predicted_probability']),
            data[['teamIDwinner', 'team_teamID']].reset_index(drop=True)
        ],
        axis=1)
    predictions_df.drop('_', 1, inplace=True)
    predictions_df.rename(columns={'team_teamID': 'team', 'teamIDwinner': 'winner'}, inplace=True)
    predictions_df['winner'] = np.where(predictions_df['winner'] == 0, 'no', 'yes')
    return predictions_df.to_dict(orient='records')


def get_current_timestamp():
    return datetime.datetime.now()
