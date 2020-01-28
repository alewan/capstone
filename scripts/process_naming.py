# Created by Aleksei Wan on 13.11.2019
# Description: Helper file for RAVDESS Name Processing

# Imports
import re

RAVDESS_STRING = r'(\d+)-(\d+)-(\d+)-(\d+)-(\d+)-(\d+)-(\d+)'
RAVDESS_NAME = re.compile(RAVDESS_STRING)
RAVDESS_FILE = re.compile(RAVDESS_STRING + '.mp4$')

RAVDESS_EMOTION_MAPPING = ['NEUTRAL', 'CALM', 'HAPPY', 'SAD', 'ANGRY', 'FEARFUL', 'DISGUST', 'SURPRISED']
AWS_RAVDESS_LIST = ['NEUTRAL', 'CALM', 'HAPPY', 'SAD', 'ANGRY', 'FEAR', 'DISGUSTED', 'SURPRISED']
AWS_EMOTION_DICT = {'NEUTRAL': 0, 'CALM': 1, 'HAPPY': 2, 'SAD': 3, 'ANGRY': 4, 'FEAR': 5, 'DISGUSTED': 6,
                    'SURPRISED': 7, 'CONFUSED': 8}

RAVDESS_NUM_ACTORS = 24
CREMA_NUM_ACTORS = 0


def is_ravdess_file(file_name: str) -> bool:
    return re.match(RAVDESS_FILE, file_name) is not None


def is_ravdess_name(name: str) -> bool:
    return re.match(RAVDESS_NAME, name) is not None

def is_crema_name(name: str) -> bool:
    pass

def get_emotion_from_ravdess_name(name: str) -> str:
    a = re.match(RAVDESS_NAME, name)
    if a is not None:
        idx = int(a.group(3)) - 1
        return RAVDESS_EMOTION_MAPPING[idx] if idx < len(RAVDESS_EMOTION_MAPPING) \
            else 'ERROR - ' + a.group(3) + 'out of range'
    return 'ERROR - No match found'


def get_emotion_from_ravdess_name_aws(name: str) -> str:
    a = re.match(RAVDESS_NAME, name)
    if a is not None:
        idx = int(a.group(3)) - 1
        return AWS_RAVDESS_LIST[idx] if idx < len(AWS_RAVDESS_LIST) else 'ERROR - ' + a.group(3) + 'out of range'
    return 'ERROR - No match found'


def get_emotion_num_from_ravdess_name(name: str) -> int:
    a = re.match(RAVDESS_NAME, name)
    if a is not None:
        idx = int(a.group(3)) - 1
        return idx if idx < len(RAVDESS_EMOTION_MAPPING) else -1
    return -1


def aws_number_from_emotion(emotion: str) -> int:
    try:
        return AWS_EMOTION_DICT[emotion]
    except KeyError:
        return -1


def get_actor_from_ravdess_name(name: str) -> int:
    a = re.match(RAVDESS_NAME, name)
    if a is not None:
        idx = int(a.group(7)) - 1
        return idx if idx < RAVDESS_NUM_ACTORS else -1
    return 'ERROR - No match found'


def get_actor_from_crema_name(name: str) -> int:
    pass


