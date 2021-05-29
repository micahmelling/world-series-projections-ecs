import joblib

from zipfile import ZipFile
from ds_helpers import aws

from modeling.train import assemble_modeling_data
from modeling.config import TARGET


def main():
    df = assemble_modeling_data()
    y = df[TARGET]
    x = df.drop(TARGET, 1)
    original_model = joblib.load('original_model.pkl')
    retrained_model = original_model.fit(x, y)
    joblib.dump(retrained_model, 'model.pkl')
    with ZipFile('sample.zip', 'w') as zip_file:
        zip_file.write('model.pkl')
    aws.upload_file_to_s3('model.zip', 'world-series-predictions-model')


if __name__ == "__main__":
    main()
