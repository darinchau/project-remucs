# Uploads spectrograms to GCP bucket
from google.cloud import storage
from google.oauth2 import service_account
from tqdm.auto import tqdm
import os

UPLOADED_FILES = "./resources/dataset/uploaded_files.txt"

def upload_files(bucket_name, source_folder, credentials_path):
    """Upload files to GCP bucket."""
    # Load credentials from the service account file
    credentials = service_account.Credentials.from_service_account_file(credentials_path)

    # Initialize the Google Cloud client with the credentials
    storage_client = storage.Client(credentials=credentials)
    bucket = storage_client.bucket(bucket_name)

    if not os.path.exists(UPLOADED_FILES):
        with open(UPLOADED_FILES, "w") as f:
            f.write("")

    root_files = [blob.name for blob in bucket.list_blobs() if '/' not in blob.name]
    root_files = set(root_files)
    print(f"Found {len(root_files)} files in the bucket")

    files = os.listdir(source_folder)
    for file_name in tqdm(files):
        try:
            if file_name in root_files:
                print(f"{file_name} already uploaded. Skipping...")
                continue
            local_path = os.path.join(source_folder, file_name)
            if os.path.isfile(local_path):
                blob = bucket.blob(file_name)
                if blob.name in root_files:
                    tqdm.write(f"{file_name} already exists in the bucket. Skipping...")
                    continue
                blob.upload_from_filename(local_path)
                tqdm.write(f"Uploaded {file_name} to {bucket_name}")
        except Exception as e:
            print(f"Error uploading {file_name}: {e}")

if __name__ == "__main__":
    bucket_name = 'project-remucs-spectrograms-1'  # replace with your bucket name
    source_folder = "D:/Repository/project-remucs/audio-infos-v3/spectrograms"  # the local folder containing files to upload
    credentials_path = "./resources/key/key.json"  # the path to the service account file

    upload_files(bucket_name, source_folder, credentials_path)
