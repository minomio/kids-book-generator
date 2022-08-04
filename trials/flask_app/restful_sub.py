from google.cloud import storage
import os

# # https://cloud.google.com/storage/docs/samples/storage-set-metadata#storage_set_metadata-python


def upload_pic(prompt):
    keyFile = "./secret/key.json"
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"]= keyFile


    bucket_name = "et-test-bucket"
    source_file_name = "./IMG_6F1F66482A84-1.jpeg"
    destination_blob_name = "flaskTrial/image_with_metadata.jpeg"

    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(destination_blob_name)
    blob.metadata = {'prompt': prompt}
    blob.upload_from_filename(source_file_name)

    print(
        f"File {source_file_name} uploaded to {destination_blob_name}."
    )

def print_something():
    return ("this is printing something")

if __name__ == "__main__":
    upload_pic("This is my phone")
