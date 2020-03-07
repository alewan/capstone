#! python

# imports 
import os

# function imports for pipeline components
from single_file_proc import create_av_files_from_input
from use_nn import audio_neural_network
from image_processing_aws import image_processing_aws
from prep_data_for_booster import prep_data_for_booster
from use_lightgbm_tree import lightgbm_tree


# this script combines the following components:
# 1: .mp4 pre-processing
# 2: audio neural network
# 3. AWS image classification 
# 4. LightGMB data preparation
# 5. LightGMB tree classification
# returns: emotion prediction

def run_pipeline():
    FINAL_EMPTION_MAPPING = ['NEUTRAL', 'CALM', 'HAPPY', 'SAD', 'ANGRY', 'FEARFUL', 'DISGUST', 'SURPRISED']

    # 1: .mp4 pre-processing

    input_file_path = "pipeline_helper/user_upload/sample_input_file.mp4"

    create_av_files_from_input(input_file_path)

    # 2: audio neural network

    checkpoint = "../pipeline/pipeline_helper/audio_nn_checkpoint"
    audio_dir = "../pipeline/pipeline_helper/audio_preprocessed/"
    batch_size = 100

    audio_neural_network(checkpoint, audio_dir, batch_size)

    # 3: AWS image classification 

    input_dir = "../pipeline/pipeline_helper/images_preprocessed/"
    output_file = "../pipeline/pipeline_helper/aws_results.json"

    image_processing_aws(input_dir, output_file)

    # 4. LightGMB data perparation 

    audio_input_file = "../pipeline/pipeline_helper/audio_nn_predictions.csv"
    audio_names_file = "../pipeline/pipeline_helper/audio_nn_prediction_names.csv"
    image_input_file = "../pipeline/pipeline_helper/aws_results.json"
    out_file = "../pipeline/pipeline_helper/results_for_lgbm.json"

    prep_data_for_booster(audio_input_file, audio_names_file, image_input_file, out_file)

    # 5. LightGMB tree classification

    input_file = "../pipeline/pipeline_helper/results_for_lgbm.json"
    model = "../pipeline/pipeline_helper/lgbm-model.txt"

    classification = lightgbm_tree(input_file, model)

    # get emotion from prediction
    emotion = FINAL_EMPTION_MAPPING[classification[0]]

    # delete everything at the end!

    trash_path = "../pipeline/pipeline_helper/audio_preprocessed"
    files = os.listdir(trash_path)
    for filename in files:
        file_path = os.path.join(trash_path, filename)
        os.remove(file_path)

    trash_path = "../pipeline/pipeline_helper/images_preprocessed"
    files = os.listdir(trash_path)
    for filename in files:
        file_path = os.path.join(trash_path, filename)
        os.remove(file_path)

    os.remove("../pipeline/pipeline_helper/results_for_lgbm.json")
    os.remove("../pipeline/pipeline_helper/aws_results.json")
    os.remove("../pipeline/pipeline_helper/audio_nn_prediction_names.csv")
    os.remove("../pipeline/pipeline_helper/audio_nn_predictions.csv")

    # return emotion prediction
    print(emotion)
    return emotion


if __name__ == "__main__":
    run_pipeline()
