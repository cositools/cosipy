import subprocess, os

def fetch_wasabi_file(file,
                      output = None,
                      override = False,
                      bucket = 'cosi-pipeline-public',
                      endpoint = 'https://s3.us-west-1.wasabisys.com',
                      access_key_id = 'GBAL6XATQZNRV3GFH9Y4',
                      access_key = 'GToOczY5hGX3sketNO2fUwiq4DJoewzIgvTCHoOv'):

    if output is None:
        output = file.split('/')[-1]

    if os.path.exists(output) and not override:
        raise RuntimeError(f"File {output} already exists.")
        
    subprocess.run(['aws', 's3api', 'get-object',
                    '--bucket', bucket,
                    '--key', file,
                    '--endpoint-url', endpoint,
                    output], 
                   env = os.environ.copy() | {'AWS_ACCESS_KEY_ID':access_key_id,
                                              'AWS_SECRET_ACCESS_KEY':access_key})
