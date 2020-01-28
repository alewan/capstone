#! python

# Written by Aleksei Wan and Meghan Muldoon on 25.10.2019

from argparse import ArgumentParser
import os
import subprocess
from python_speech_features import mfcc, logfbank
import scipy.io.wavfile as wav
import wave


# THINGS THAT THIS WILL DO:
# 1) CROP VIDEO TO REMOVE SILENCE
# 2) TURN INTO A WAV FILE
# 3) SAMPLE THE VIDEO AND AUDIO (0.668 ms)
# 4) VECTORIZE WAV FILE AND FIND MFCC


def create_av_files_from_input(dir_path: str):
    if not os.path.isdir(dir_path):
        print("Directory DNE")
        return "Directory doesn't exist"

    # Files = a list of filenames in the current directory
    files = os.listdir(dir_path)

    # Manually remove hidden files (.DS_Store) from the filelist
    # Don't use glob to remove hidden files;
    files_no_hidden = []
    for filename in files:
        if not filename.startswith('.'):
            files_no_hidden.append(filename)

    # Go through all the files in the folder
    for filename in files_no_hidden:
        file_path = os.path.join(dir_path, filename)

        new_folder_path = file_path + "_proc_data"
        os.mkdir(new_folder_path)

        # Raw name without mp4 extension
        name = os.path.splitext(filename)[0]

        # Hard coded output names
        # output_name =  "/Users/meghanmuldoon/Desktop/jpg_dump2/" + filename
        aud_output_name = os.path.join(os.path.expanduser("~"), 'Desktop', 'audio_dump', name)

        output_name = new_folder_path + "/" + name
        wav_file_name = output_name + '.wav'
        silrem_wav_out = new_folder_path + "/sil_rem.wav"
        mono_file_name = output_name + 'mono.wav'

        command1 = "(ffmpeg -i " + file_path + " -vn " + wav_file_name + ")"
        subprocess.call(command1, shell=True)

        # COMMAND TO SAMPLE THE FAMES
        # Sample = 5/3 fps (Sample every 0.6 seconds)
        command2 = "(ffmpeg -i " + file_path + " -vf fps=5/3 " + output_name + "_%04d.jpg -hide_banner) "
        subprocess.call(command2, shell=True)

        # Convert wav to mono wav aformat=dblp,areverse
        command3 = "(ffmpeg -i " + wav_file_name + " -ac 1 " + mono_file_name + ")"
        subprocess.call(command3, shell=True)

        # Remove the silence
        command4 = "(ffmpeg -i " + mono_file_name + " -af silenceremove=1:0:-50dB:stop_periods=1 " + silrem_wav_out + ")"
        subprocess.call(command4, shell=True)

        # Split silence removed into 0.6 second segments
        command5 = "(ffmpeg -i " + silrem_wav_out + " -f segment -segment_time 0.6 " + aud_output_name + "_%04d.wav -hide_banner) "
        subprocess.call(command5, shell=True)

        # Rate = the sampling rate of the wav file (samples/second)
        # Sig = data read as a numpy array
        (rate, sig) = wav.read(wav_file_name)

        # file is a wave_read object
        file = wave.open(wav_file_name, 'r')
        samples = file.getnframes()

        duration = samples / rate

        print(duration)

        mfcc_feat = mfcc(sig, rate)
        fbank_feat = logfbank(sig, rate)

        # print(fbank_feat[1:3,:])


if __name__ == "__main__":
    parser = ArgumentParser(description='Create jpg images and wav file from a directory of mp4 inputs')
    parser.add_argument('--input_directory', type=str, default='testvideo1.mp4', help='filepath to dir')
    args = parser.parse_args()

    create_av_files_from_input(args.input_directory)
