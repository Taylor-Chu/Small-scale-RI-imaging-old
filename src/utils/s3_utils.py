import os
import boto3
from tqdm import tqdm
from botocore.config import Config
import glob
import fsspec
from io import BytesIO


def download_file_from_s3(s3_key: str, local_path: str):
    """
    Downloads a file from S3 to a local path.

    Args:
        s3_key (str): The S3 key of the file to download.
        local_path (str): The local path where the file will be saved.
    """
    os.environ["AWS_ACCESS_KEY_ID"] = "Y7MG0DGL3A2LY1ANN5BZ"
    os.environ["AWS_SECRET_ACCESS_KEY"] = "GMCWK7aAn1ZlAIcFf8h2tYn7D2RDYdzjlxWkSYde"
    os.environ["AWS_ENDPOINT_URL"] = "https://s3.eidf.ac.uk"
    bucket_name = "r2d2data"

    config = Config(
        request_checksum_calculation="when_required",
        response_checksum_validation="when_required",
    )
    s3 = boto3.resource('s3', config=config)
    bucket = s3.Bucket(bucket_name)

    fs = fsspec.filesystem("s3")
    fs.get(s3_key, local_path)
    print(f"Downloaded {s3_key} to {local_path}")