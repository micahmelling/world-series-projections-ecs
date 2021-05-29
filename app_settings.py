from ds_helpers import aws


AWS_KEYS_SECRET_NAME = 'world-series-app-keys'
APP_SECRET = aws.get_secrets_manager_secret('world-series-app-secret').get('secret')
LOGGING_S3_BUCKET_NAME = 'world-series-predictions-app-logs'
