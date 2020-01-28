#! python

# Created by Aleksei Wan on 28.10.2019

# Imports
import os
from sys import exit
import re
from argparse import ArgumentParser
import boto3
import json

# regex for image file matching
IMG_FILE = re.compile('(.*)\.jp[e]?g$')


# Note: This depends on your AWS credentials and configs being set up
def detect_labels_local_file(photo, full_responses: bool):
    """
    :param photo: photo file
    :param full_responses: return full responses or just the emotion section
    :return: list of labels
    """
    client = boto3.client('rekognition')

    with open(photo, 'rb') as image:
        response = client.detect_faces(Image={'Bytes': image.read()}, Attributes=['ALL'])
        # sample_response = {'FaceDetails': [{'BoundingBox': {'Width': 0.23524627089500427, 'Height': 0.589076578617096, 'Left': 0.38697147369384766, 'Top': 0.1886293888092041}, 'AgeRange': {'Low': 22, 'High': 34}, 'Smile': {'Value': False, 'Confidence': 95.76689910888672}, 'Eyeglasses': {'Value': False, 'Confidence': 99.7113265991211}, 'Sunglasses': {'Value': False, 'Confidence': 99.91047668457031}, 'Gender': {'Value': 'Male', 'Confidence': 98.21485900878906}, 'Beard': {'Value': False, 'Confidence': 72.09636688232422}, 'Mustache': {'Value': False, 'Confidence': 98.3759765625}, 'EyesOpen': {'Value': True, 'Confidence': 93.06100463867188}, 'MouthOpen': {'Value': False, 'Confidence': 98.17819213867188}, 'Emotions': [{'Type': 'SURPRISED', 'Confidence': 0.1166321188211441}, {'Type': 'DISGUSTED', 'Confidence': 0.6705758571624756}, {'Type': 'ANGRY', 'Confidence': 0.9103255271911621}, {'Type': 'FEAR', 'Confidence': 0.08695628494024277}, {'Type': 'CALM', 'Confidence': 93.19449615478516}, {'Type': 'HAPPY', 'Confidence': 1.9579323530197144}, {'Type': 'SAD', 'Confidence': 2.3680579662323}, {'Type': 'CONFUSED', 'Confidence': 0.6950306296348572}], 'Landmarks': [{'Type': 'eyeLeft', 'X': 0.45723333954811096, 'Y': 0.4198251962661743}, {'Type': 'eyeRight', 'X': 0.5646100640296936, 'Y': 0.43383580446243286}, {'Type': 'mouthLeft', 'X': 0.4569289982318878, 'Y': 0.6296104192733765}, {'Type': 'mouthRight', 'X': 0.545610249042511, 'Y': 0.6410675048828125}, {'Type': 'nose', 'X': 0.5080152153968811, 'Y': 0.5402817130088806}, {'Type': 'leftEyeBrowLeft', 'X': 0.41665807366371155, 'Y': 0.36552050709724426}, {'Type': 'leftEyeBrowRight', 'X': 0.48295462131500244, 'Y': 0.36212068796157837}, {'Type': 'leftEyeBrowUp', 'X': 0.4511050581932068, 'Y': 0.34688517451286316}, {'Type': 'rightEyeBrowLeft', 'X': 0.5460655689239502, 'Y': 0.37050577998161316}, {'Type': 'rightEyeBrowRight', 'X': 0.6084395051002502, 'Y': 0.3907383680343628}, {'Type': 'rightEyeBrowUp', 'X': 0.5782649517059326, 'Y': 0.36350154876708984}, {'Type': 'leftEyeLeft', 'X': 0.43805167078971863, 'Y': 0.4152671992778778}, {'Type': 'leftEyeRight', 'X': 0.47795194387435913, 'Y': 0.4236871302127838}, {'Type': 'leftEyeUp', 'X': 0.4574522078037262, 'Y': 0.4093792736530304}, {'Type': 'leftEyeDown', 'X': 0.457261860370636, 'Y': 0.4282190203666687}, {'Type': 'rightEyeLeft', 'X': 0.5418963432312012, 'Y': 0.4320144057273865}, {'Type': 'rightEyeRight', 'X': 0.5813888907432556, 'Y': 0.4338427484035492}, {'Type': 'rightEyeUp', 'X': 0.5637412071228027, 'Y': 0.42317792773246765}, {'Type': 'rightEyeDown', 'X': 0.5620729923248291, 'Y': 0.4417969882488251}, {'Type': 'noseLeft', 'X': 0.48525646328926086, 'Y': 0.5603309273719788}, {'Type': 'noseRight', 'X': 0.5248827934265137, 'Y': 0.563277542591095}, {'Type': 'mouthUp', 'X': 0.5030224323272705, 'Y': 0.6107248663902283}, {'Type': 'mouthDown', 'X': 0.49978896975517273, 'Y': 0.6716505289077759}, {'Type': 'leftPupil', 'X': 0.45723333954811096, 'Y': 0.4198251962661743}, {'Type': 'rightPupil', 'X': 0.5646100640296936, 'Y': 0.43383580446243286}, {'Type': 'upperJawlineLeft', 'X': 0.3847735822200775, 'Y': 0.40720877051353455}, {'Type': 'midJawlineLeft', 'X': 0.39927977323532104, 'Y': 0.6344922184944153}, {'Type': 'chinBottom', 'X': 0.49391692876815796, 'Y': 0.776694118976593}, {'Type': 'midJawlineRight', 'X': 0.5928965210914612, 'Y': 0.6594374775886536}, {'Type': 'upperJawlineRight', 'X': 0.6258419752120972, 'Y': 0.4382151663303375}], 'Pose': {'Roll': 3.7603917121887207, 'Yaw': 2.560701370239258, 'Pitch': -0.901584804058075}, 'Quality': {'Brightness': 88.5201187133789, 'Sharpness': 89.85481262207031}, 'Confidence': 100.0}], 'ResponseMetadata': {'RequestId': '1c1cc047-9aa1-4618-a52a-5ff1369f656b', 'HTTPStatusCode': 200, 'HTTPHeaders': {'content-type': 'application/x-amz-json-1.1', 'date': 'Mon, 28 Oct 2019 19:33:32 GMT', 'x-amzn-requestid': '1c1cc047-9aa1-4618-a52a-5ff1369f656b', 'content-length': '3328', 'connection': 'keep-alive'}, 'RetryAttempts': 0}}

    return response if full_responses else response['FaceDetails'][0]['Emotions']


def detect_photos_in_dir(directory: str, full_responses: bool) -> list:
    """
    :param directory: directory to evaluate
    :param full_responses: return full responses or just the emotion section
    :return: list of pairs of photos and tags
    """
    labels = list()
    files = os.listdir(directory)
    num_files = str(len(files))

    print('Using', directory, 'as images directory...')
    print(num_files, 'file(s) found.')

    tracking = 0
    for photo_name in files:
        tracking += 1
        print('(' + str(tracking) + '/' + num_files + ')', end=' ')
        img_file = re.match(IMG_FILE, photo_name)
        if img_file:
            print('Evaluating image file', img_file[1])
            photo = os.path.join(directory, photo_name)
            labels.append((img_file[1], detect_labels_local_file(photo, full_responses)))
        else:
            print('Ignoring non-image file', photo_name)

    return labels


if __name__ == "__main__":
    parser = ArgumentParser(description='Send images to AWS for processing')
    parser.add_argument('--input_dir', type=str, default='images', help='Dir containing images for classification')
    parser.add_argument('--output_file', type=str, help='File to cache results', required=False)
    args = parser.parse_args()

    path_to_check = os.path.abspath(args.input_dir)

    get_full_responses = args.output_file if args.output_file is not None else False

    if not os.path.exists(path_to_check):
        print('Provided path', path_to_check, 'is not a valid directory. Please try again')
        exit(-1)

    photo_labels = detect_photos_in_dir(path_to_check, get_full_responses)

    if get_full_responses:
        with open(args.output_file, 'w+') as f:
            json.dump(photo_labels, f)
    else:
        print(photo_labels)
