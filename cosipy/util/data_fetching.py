import boto3
from hashlib import md5
from pathlib import Path
import zipfile
import gzip
import logging

logger = logging.getLogger(__name__)

def fetch_wasabi_file(file,
                      output = None,
                      overwrite = False,
                      unzip = False,
                      unzip_output = None,
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
    overwrite: bool, optional
        Whether to overwrite the file or throw an error if the file already exists
        and has as different checksum. If the file exists but has the same checksum
        it will always throw a warning and skip the file.
    unzip: bool, optional
        Uncompress a .gz or .zip file.
    unzip_output: str, Path, optional
        Path to unzipped output, if different from output without .gz or .zip
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

    Returns
    -------
    dict containing file metadata
    """

    s3 = boto3.client('s3',
                      endpoint_url=endpoint,
                      aws_access_key_id=access_key,
                      aws_secret_access_key=secret_key)

    file_hdr = s3.head_object(Bucket=bucket, Key=file)

    if output is None:
        output = file.split('/')[-1]

    output = Path(output)

    if unzip:

        zip_suffix = output.suffix
        if zip_suffix not in ['.gz', '.zip']:
            raise ValueError(f"Only the compression of of .zip and .gz files is supported. Got a '{zip_suffix}' file")

        if unzip_output is None:
            unzip_output = output.with_suffix('')
        else:
            unzip_output = Path(unzip_output)

        if unzip_output.exists():

            if checksum is None:

                if overwrite:
                    logger.warning(f"A file named {unzip_output} already exists, but checksum was not provided. Will override.")
                else:
                    raise RuntimeError(
                        f"A file named {unzip_output} already exists, override=False, and checksum was not provided.")

            else:

                local_checksum = md5(open(unzip_output, 'rb').read()).hexdigest()

                if local_checksum != checksum:
                    if overwrite:
                        logger.warning(f"A file named {unzip_output} already exists but has a different checksum ({local_checksum}) than specified ({checksum}). Will override.")
                        unzip_output.unlink() #Delete
                    else:
                        raise RuntimeError(f"A file named {unzip_output} already exists but has a different checksum ({local_checksum}) than specified ({checksum}).")
                else:
                    logger.warning(f"A file named {unzip_output} already exists with the specified checksum ({checksum}). Skipping.")
                    return file_hdr

        # Get zipped file
        fetch_wasabi_file(file, output = output, overwrite= overwrite, unzip= False, bucket = bucket, endpoint = endpoint, access_key = access_key, secret_key = secret_key)

        # Unzip
        if zip_suffix == '.zip':
            with zipfile.ZipFile(output, 'r') as f_in:

                file_list = f_in.infolist()

                if len(file_list) > 1 or ('/' in file_list[0].filename):
                    # Multiple files requires tracking them and checking multiple checksums. Let's keep it simple for now.
                    raise RuntimeError("We currently only support unzipping files containing a single file and no folders.")

                f_in.extractall(output.parent)
        elif zip_suffix == '.gz':
            with gzip.open(output, 'rb') as f_in, open(unzip_output, 'wb') as f_out:
                f_out.write(f_in.read())

        return file_hdr

    if output.exists():

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
            remote_etag = file_hdr["ETag"][1:-1]  # Remove quotes

            # Compare sizes
            remote_size = file_hdr['ContentLength']
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

        if (remote_size != local_size) or remote_etag != local_etag:
            if overwrite:
                logger.warning(f"A file named {output} already exists but has a different checksum than the requested file ({remote_etag}). Will override.")
            else:
                raise RuntimeError(f"A file named {output} already exists but has a different checksum than the requested file ({remote_etag})")
        else:
            logger.warning(f"A file named {output} with the same ETag ({remote_etag}) as the requested file already exists. Skipping.")
            return file_hdr

    logger.info(f"Downloading {bucket}/{file} ({file_hdr['ContentLength']} bytes)")
    s3.download_file(Bucket=bucket, Key=file, Filename=output)

    return file_hdr

def fetch_wasabi_file_header(file,
                             bucket = 'cosi-pipeline-public',
                             endpoint = 'https://s3.us-west-1.wasabisys.com',
                             access_key = 'GBAL6XATQZNRV3GFH9Y4',
                             secret_key = 'GToOczY5hGX3sketNO2fUwiq4DJoewzIgvTCHoOv'):
    """
    Get the metadata for a file from COSI's Wasabi acccount.

    Parameters
    ----------
    file : str
        Full path to file in Wasabi
    bucket : str, optional
        Passed to aws --bucket option
    endpoint : str, optional
        Passed to aws --endpoint-url option
    access_key : str, optional
        AWS_ACCESS_KEY_ID
    secret_key : str, optional
        AWS_SECRET_ACCESS_KEY

    Returns
    -------
    dict containing file metadata
    """

    s3 = boto3.client('s3',
                      endpoint_url=endpoint,
                      aws_access_key_id=access_key,
                      aws_secret_access_key=secret_key)

    return s3.head_object(Bucket=bucket, Key=file)
