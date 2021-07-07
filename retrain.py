import joblib
import os

from zipfile import ZipFile
from ds_helpers import aws

from modeling.train import assemble_modeling_data
from modeling.config import TARGET


def main():
    df = assemble_modeling_data()
    y = df[TARGET]
    x = df.drop(TARGET, 1)
    aws.download_file_from_s3('model.zip', 'world-series-predictions-model')
    os.makedirs('original_model')
    with ZipFile('model.zip', 'r') as zip_file:
        zip_file.extractall('original_model')
    original_model = joblib.load(os.path.join('original_model', 'model.pkl'))
    retrained_model = original_model.fit(x, y)
    joblib.dump(retrained_model, 'model.pkl')
    with ZipFile('model.zip', 'w') as zip_file:
        zip_file.write('model.pkl')
    aws.upload_file_to_s3('model.zip', 'world-series-predictions-model')


if __name__ == "__main__":
    main()
