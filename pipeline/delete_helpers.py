#! python

# imports 
import os

def delete_helpers():

   # begin by deleting everything

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
   
    os.remove("../pipeline/pipeline_helper/aws_results.json")
    os.remove("../pipeline/pipeline_helper/audio_nn_prediction_names.csv")
    os.remove("../pipeline/pipeline_helper/audio_nn_predictions.csv")
    os.remove("../pipeline/pipeline_helper/audio_nn_predictions.json")
    os.remove("../pipeline/pipeline_helper/results_for_lgbm.json")
    os.remove("../pipeline/pipeline_helper/user_upload/input_file.mp4")

if __name__ == "__main__":
    delete_helpers()