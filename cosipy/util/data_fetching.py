import os

import boto3
from hashlib import md5
from pathlib import Path
import zipfile
import gzip
import logging
logger = logging.getLogger(__name__)

def fetch_wasabi_file(file,
                      output = None,
                      override = False,
                      unzip = False,
                      checksum = None,
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
    unzip: bool, optional
        Uncompress a .gz or .zip file.
    checksum: str
        Expected MD5 sum (hex string) of the uncompressed file. Only used when unzip=True.
        The checksum of the original file in wasabi is automatically obtained from the server.
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

    if unzip:

        zip_suffix = output.suffix

        if zip_suffix not in ['.zip', '.gz']:
            raise ValueError(f"Only the compression of of .zip and .gz files is supported. Got a '{zip_suffix}' file")

        unzip_output = output.with_suffix('')

        if unzip_output.exists():
            if override:
                unzip_output.unlink() # Delete
            else:

                if checksum is None:
                    raise RuntimeError(f"A file named {unzip_output} already exists, override=False, and checksum was not provided.")

                local_checksum = md5(open(unzip_output, 'rb').read()).hexdigest()

                if local_checksum != checksum:
                    raise RuntimeError(f"A file named {unzip_output} already exists but has a different checksum ({local_checksum}) than specified ({checksum}).")
                else:
                    logger.warning(f"A file named {unzip_output} already exists with the specified checksum ({checksum}). Skipping.")
                    return

        # Get zipped file
        fetch_wasabi_file(file, output = output, override = override, unzip= False, bucket = bucket, endpoint = endpoint, access_key = access_key, secret_key = secret_key)

        # Unzip
        if zip_suffix == '.zip':
            with zipfile.ZipFile(output, 'r') as f_in:
                f_in.extractall(output.parent)
        elif zip_suffix == '.gz':
            with gzip.open(output, 'rb') as f_in, open(unzip_output, 'wb') as f_out:
                f_out.write(f_in.read())

        return

    if output.exists() and not override:

        def get_size_and_etag():
            """
            AWS/Wasabi API do not have a way to retrieve the MD5 check sum for a file,
            unless it was provided during the upload.

            They have something called the "ETag". For small file, it is the same
            as the MD5 sum. However, large file are usualy splitted in part, and the
            ETag is computed by getting the MD5 of the concatenation of the MD5 sum of the
            individual parts, with the suffix `-N` where N is the number of parts.
            I took the code to do this calculations from here: https://teppen.io/2018/06/23/aws_s3_etags/

            Thowever, that code requires part size, which had to be guess. I modified it
            by querying the part sizes, instead of guessing them, following this answer:
            https://stackoverflow.com/a/63271381

            Once the ETag for the local file is computed, the Wasabi file and output file
            are considered equal is they have the same ETag

            Returns:
                (remote_size, local_size),(remote_etag, local_etag)
                if local_size != remote_size, the local etag can't be computed and is None
            """

            # Get header information
            remote_head = s3.head_object(Bucket=bucket, Key=file)

            remote_etag = remote_head["ETag"][1:-1]  # Remove quotes

            # Compare sizes
            remote_size = remote_head['ContentLength']
            local_size = output.stat().st_size

            if remote_size != local_size:
                # We already know they are not the same size
                # The actual etag can't be computed, since we
                # don't know the part size. Return MD5
                return (remote_size, local_size),(remote_etag, None)

            # File have the same size
            etag_parts = remote_etag.split("-")

            if len(etag_parts) == 1:
                # Not divided in parts. etag = md5 sum
                local_etag = md5(open(output, 'rb').read()).hexdigest()
            else:
                # Divided in parts. It can be a single part.
                nparts = int(etag_parts[1])

                # Determine part sizes
                part_sizes = []

                for part in range(1, nparts + 1): # 1-index
                    out = s3.head_object(Bucket=bucket, Key=file, PartNumber=part)

                    part_sizes += [out['ContentLength']]

                # Get md5 sums
                md5_digests = []
                with open(output, 'rb') as f:
                    for part_size in part_sizes:
                        chunk = f.read(part_size)
                        md5_digests.append(md5(chunk).digest())
                local_etag = md5(b''.join(md5_digests)).hexdigest() + '-' + str(nparts)

            return (remote_size, local_size),(remote_etag, local_etag)

        (remote_size, local_size), (remote_etag, local_etag) = get_size_and_etag()

        if remote_size != local_size:
            raise RuntimeError(f"A file named {output} already exists but has the wrong file size ({local_size}) than the requested file ({remote_size}).")
        elif remote_etag != local_etag:
            raise RuntimeError(f"A file named {output} already exists but has a different ETag ({local_etag}) than the requested file ({remote_etag}).")
        else:
            logger.warning(f"A file named {output} with the same ETag ({remote_etag}) as the requested file already exists. Skipping.")
            return

    s3.download_file(Bucket=bucket, Key=file, Filename=output)

