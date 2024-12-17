import boto3
import os

def download_s3_folder(bucket_name, prefix, local_path):
    AWS_AK="----"
    AWS_SK="----"
    s3 = boto3.client( 
        's3',
        aws_access_key_id=AWS_AK,
        aws_secret_access_key=AWS_SK
    )

    response = s3.list_objects_v2(
        Bucket=bucket_name,
        Prefix=prefix
    )

    while True:
        for obj in response['Contents']:
            local_file_path = os.path.join(local_path, obj['Key'].replace(prefix, ''))
            os.makedirs(os.path.dirname(local_file_path), exist_ok=True)
            s3.download_file(bucket_name, obj['Key'], local_file_path)

        if 'NextContinuationToken' in response:
            response = s3.list_objects_v2(Bucket=bucket_name,
                                           Prefix=prefix,
                                           ContinuationToken=response['NextContinuationToken'])
        else:
            break

# Example usage:
bucket_name = "----"
prefix = 'model/'
local_path = '----'

download_s3_folder(bucket_name, prefix, local_path)