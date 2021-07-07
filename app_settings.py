import os

from ds_helpers import aws


AWS_KEYS_SECRET_NAME = 'world-series-app-keys'
APP_SECRET = aws.get_secrets_manager_secret('world-series-app-secret').get('secret')
LOGGING_S3_BUCKET_NAME = 'world-series-predictions-app-logs'
ENVIRONMENT = os.environ['ENVIRONMENT']

if ENVIRONMENT in ['local', 'stage']:
    DATABASE_SECRET = 'stage-world-series-svc-mysql'
elif ENVIRONMENT == 'prod':
    DATABASE_SECRET = 'prod-world-series-svc-mysql'
else:
    raise Exception(f'ENVIRONMENT must be one of local, stage, or prod. {ENVIRONMENT} was passed.')
