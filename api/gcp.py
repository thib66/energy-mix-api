from google.cloud import storage
from google.oauth2 import service_account
from dotenv import load_dotenv

import os

load_dotenv()

GOOGLE_APPLICATION_CREDENTIALS = os.environ.get('GOOGLE_APPLICATION_CREDENTIALS', '')
GCP_BUCKET = os.environ.get('GCP_BUCKET', '')
google_credentials = service_account.Credentials.from_service_account_file(GOOGLE_APPLICATION_CREDENTIALS)

def gcp_load_model():

    client = storage.Client(credentials=google_credentials)

    # Accéder au bucket
    bucket = client.bucket(GCP_BUCKET)

    file_name="bi-lstm_model_V1.h5"
    # Télécharger le fichier dans un objet local (par exemple dans un fichier temporaire)
    blob = bucket.blob(file_name)

    # Sauvegarder le fichier h5 localement pour l'ouvrir avec h5py
    local_filename = file_name  # Emplacement temporaire pour le fichier téléchargé
    blob.download_to_filename(local_filename)
                              
    print(f"Le fichier {file_name} a été téléchargé avec succès depuis le bucket {bucket.name}.")

    return local_filename


def gcp_load_data():
    client = storage.Client(credentials=google_credentials)

    # Accéder au bucket
    bucket = client.bucket(GCP_BUCKET)

    file_name="dataset_final.csv"
    # Télécharger le fichier dans un objet local (par exemple dans un fichier temporaire)
    blob = bucket.blob(file_name)

    local_filename = file_name  # Emplacement temporaire pour le fichier téléchargé
    blob.download_to_filename(local_filename)
                              
    print(f"Le fichier {file_name} a été téléchargé avec succès depuis le bucket {bucket.name}.")

    return local_filename


if __name__ == '__main__':
    gcp_load_data()