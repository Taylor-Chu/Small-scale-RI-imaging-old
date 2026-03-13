import os
import boto3
from tqdm import tqdm
from botocore.config import Config
import glob
import fsspec
from io import BytesIO

bucket_name = "r2d2data"

config = Config(
    request_checksum_calculation="when_required",
    response_checksum_validation="when_required",
)
s3 = boto3.resource('s3', config=config)
bucket = s3.Bucket(bucket_name)

fs = fsspec.filesystem("s3")

def download_file_from_s3(s3_key: str, local_path: str):
    """
    Downloads a file from S3 to a local path.

    Args:
        s3_key (str): The S3 key of the file to download.
        local_path (str): The local path where the file will be saved.
    """
    bucket.download_file(s3_key, local_path)