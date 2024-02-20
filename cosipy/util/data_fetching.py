import os
from awscli.clidriver import create_clidriver

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

    if os.path.exists(output) and not override:
        raise RuntimeError(f"File {output} already exists.")

    cli = create_clidriver()

    cli.session.set_credentials(access_key, secret_key)
    
    cli.main(['s3api', 'get-object',
              '--bucket', bucket,
              '--key', file,
              '--endpoint-url', endpoint,
              output])

