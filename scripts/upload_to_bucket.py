#! python

# Created by Aleksei Wan on 28.10.2019

# Imports
import os
from sys import exit
import re
from argparse import ArgumentParser
import boto3
import json

import logging
from botocore.exceptions import ClientError

# regex for image file matching
IMG_FILE = re.compile('(.*)\.jp[e]?g$')

def upload_file(file_name, bucket, object_name=None):
    """Upload a file to an S3 bucket

    :param file_name: File to upload
    :param bucket: Bucket to upload to
    :param object_name: S3 object name. If not specified then file_name is used
    :return: True if file was uploaded, else False
    """

    # If S3 object_name was not specified, use file_name
    if object_name is None:
        object_name = file_name

    # Upload the file
    s3_client = boto3.client('s3')
    try:
        response = s3_client.upload_file(file_name, bucket, object_name)
    except ClientError as e:
        logging.error(e)
        return False
    return True

# Note: This depends on your AWS credentials and configs being set up

def detect_photos_in_dir(directory: str, full_responses: bool, bucket) -> list:
    """
    :param directory: directory to evaluate
    :param full_responses: return full responses or just the emotion section
    :return: list of pairs of photos and tags
    """
    labels = list()
    files = os.listdir(directory)
    tracking = 0

    for photo_name in files:
        tracking += 1
        print(str(tracking))
        img_file = re.match(IMG_FILE, photo_name)
        if img_file:
            path = directory + "/" + photo_name
            upload_file(path, bucket)
        


if __name__ == "__main__":
    parser = ArgumentParser(description='Send images to AWS for processing')
    parser.add_argument('--input_dir', type=str, default='images', help='Dir containing images for classification')
    parser.add_argument('--output_file', type=str, help='File to cache results', required=False)
    args = parser.parse_args()

    path_to_check = os.path.abspath(args.input_dir)

    get_full_responses = args.output_file if args.output_file is not None else False

    # Retrieve the list of existing buckets
    s3 = boto3.client('s3')
    response = s3.list_buckets()

    # Output the bucket names
    print('Existing buckets:')
    for bucket in response['Buckets']:
        photo_labels = detect_photos_in_dir(path_to_check, get_full_responses, bucket["Name"])
        print(f'  {bucket["Name"]}')

    if not os.path.exists(path_to_check):
        print('Provided path', path_to_check, 'is not a valid directory. Please try again')
        exit(-1)

    photo_labels = detect_photos_in_dir(path_to_check, get_full_responses, )

    

