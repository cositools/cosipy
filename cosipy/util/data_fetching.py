import os

from IPython.utils.coloransi import value
from awscli.clidriver import create_clidriver
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
        the current durectory and the same file name as the input file.
    bucket : str, optional
        Passed to aws --bucket option
    endpoint : str, optional
        Passed to aws --endpoint-url option
    access_key : str, optional
        AWS_ACCESS_KEY_ID
    secret_key : str, optional
        AWS_SECRET_ACCESS_KEY
    """
    
    if output is None:
        output = file.split('/')[-1]

    output = Path(output)    
        
    if output.exists():
        if override is False:
            raise RuntimeError(f"File {output} already exists.")
        elif override == 'skip':
            logger.warning(f"File {output} already exists. Skipping.")
            return
        elif override is not True:
            logger.warning(f"File {output} already exists. Overriding.")
            raise ValueError(f"Parameter override can only be True, False or 'skip'. Got {override}")

    cli = create_clidriver()

    cli.session.set_credentials(access_key, secret_key)
    command = ['s3api', 'get-object',
               '--bucket', bucket,
               '--key', file,
               '--endpoint-url', endpoint,
               str(output)]

    cli.main(command)

