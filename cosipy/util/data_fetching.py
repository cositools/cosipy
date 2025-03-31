import os

import boto3
from hashlib import md5
from pathlib import Path
import logging
logger = logging.getLogger(__name__)

def fetch_wasabi_file(file,
                      output = None,
                      override = False,
                      bucket = 'cosi-pipeline-public',
                      endpoint = 'https://s3.us-west-1.wasabisys.com',
                      access_key = 'GBAL6XATQZNRV3GFH9Y4',
                      secret_key = 'GToOczY5hGX3sketNO2fUwiq4DJoewzIgvTCHoOv'):
    """
    Download a file from COSI's Wasabi acccount.

    Parameters
    ----------
    file : str
        Full path to file in Wasabi
    output : str,  optional
        Full path to the downloaded file in the local system. By default it will use 
        the current directory and the same file name as the input file.
    override: bool, optional
        If True, it will override the output file if already exists. Otherwise, it will
        throw and error, unless the existing file has the same checksum, in which case
        it will simply skip it with a warning.
    bucket : str, optional
        Passed to aws --bucket option
    endpoint : str, optional
        Passed to aws --endpoint-url option
    access_key : str, optional
        AWS_ACCESS_KEY_ID
    secret_key : str, optional
        AWS_SECRET_ACCESS_KEY
    """

    s3 = boto3.client('s3',
                      endpoint_url=endpoint,
                      aws_access_key_id=access_key,
                      aws_secret_access_key=secret_key)

    if output is None:
        output = file.split('/')[-1]

    output = Path(output)    
        
    if output.exists() and not override:

        existing_md5sum = md5(open(output, 'rb').read()).hexdigest()
        wasabi_md5sum = s3.head_object(Bucket=bucket, Key=file)["ETag"][1:-1]

        if existing_md5sum == wasabi_md5sum:
            logger.warning(f"A file named {output} already exists and has same checksum ({wasabi_md5sum}) as the requested file. Skipping.")
            return
        else:
            raise RuntimeError(f"A file named {output} already exists and has a different checksum ({existing_md5sum}) than the requested file ({wasabi_md5sum}).")

    s3.download_file(Bucket=bucket, Key=file, Filename=output)

