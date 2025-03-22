from google.cloud import storage
from google.oauth2 import service_account
from dotenv import load_dotenv

import os

load_dotenv()

GOOGLE_APPLICATION_CREDENTIALS = os.environ.get('GOOGLE_APPLICATION_CREDENTIALS', '')
GCP_BUCKET = os.environ.get('GCP_BUCKET', '')
google_credentials = service_account.Credentials.from_service_account_file(GOOGLE_APPLICATION_CREDENTIALS)

def load_model():

    client = storage.Client(credentials=google_credentials)
    print(client.project)
    bucket = client.bucket(GCP_BUCKET)
    print(bucket.name)



    return


if __name__ == '__main__':
    load_model()