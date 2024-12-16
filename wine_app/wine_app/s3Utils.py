import boto3
import os
from io import BytesIO


# --------- SETTINGS ---------
AWS_AK="ABC"
AWS_SK="ABC"

modelPath="ABC"

csvFile="TrainingDataset.csv"

s3Bucket="ABC"
fileKey=f"data/{csvFile}"
s3ModelPath="model"
# --------- SETTINGS ---------

def folderExists(bucket, remote_folder):
  #Setup S3 client
  s3 = boto3.client(
    's3',
    aws_access_key_id=AWS_AK,
    aws_secret_access_key=AWS_SK
    )
  
  #Remove if folder exists
  response = s3.list_objects_v2(Bucket=bucket, Prefix=remote_folder)
  try:
    exists = len(response['Contents']) > 0
    return True

  except KeyError as ke:
    return False

def deleteS3Folder(bucket, targetFolder):
  #Setup S3 client
  s3 = boto3.client(
    's3',
    aws_access_key_id=AWS_AK,
    aws_secret_access_key=AWS_SK
    )
  
  try:
    #Delete Folder in S3 bucket
    response = s3.list_objects_v2(Bucket=bucket, Prefix=targetFolder)

    while True:
        for obj in response['Contents']:
            s3.delete_object(Bucket=bucket, Key=obj['Key'])
        if not response['NextContinuationToken']:
            break
        response = s3.list_objects_v2(Bucket=bucket, Prefix=targetFolder, ContinuationToken=response['NextContinuationToken'])

  except Exception as ex:
    # return None
    print(str(ex))
    exit(1)



def downloadFile(targetFile, bucket, bucketKey):
  #Setup S3 client
  s3 = boto3.client(
    's3',
    aws_access_key_id=AWS_AK,
    aws_secret_access_key=AWS_SK
    )
  
  try:
    #Download locally
    s3.download_file(bucket, bucketKey, targetFile)

  except Exception as ex:
    print(str(ex))
    exit(1)


def uploadModel(bucket, local_folder, remote_folder):
  #Setup S3 client
  s3 = boto3.client(
    's3',
    aws_access_key_id=AWS_AK,
    aws_secret_access_key=AWS_SK
    )
  
  try:
    #Upload Model to S3 bucket
    for root, dirs, files in os.walk(local_folder):
        for file in files:
            local_path = os.path.join(root, file)
            relative_path = os.path.relpath(local_path, local_folder)
            s3_path = os.path.join(remote_folder, relative_path)

            s3.upload_file(local_path, bucket, s3_path)

  except Exception as ex:
    print(ex)
    exit(1)
