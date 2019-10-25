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


def create_av_files_from_input(file_name: str):
    if not os.path.isfile(file_name):
        return "File doesn't exist"

    name = os.path.splitext(file_name)[0]
    wav_file_name = name + '.wav'

    # COMMAND TO GET A WAV FILE FROM MP4
    command1 = "(ffmpeg -i " + file_name + " -vn " + wav_file_name + ")"
    subprocess.call(command1, shell=True)

    # COMMAND TO SAMPLE THE FAMES
    # Sample = 5/3 fps (Sample every 0.6 seconds)
    command2 = "(ffmpeg -i " + file_name + " -vf fps=5/3 " + name + "_%04d.jpg -hide_banner) "
    subprocess.call(command2, shell=True)

    # Rate = the sampling rate of the wav file (samples/second)
    # Sig = data read as a numpy array
    (rate, sig) = wav.read(wav_file_name)

    # file is a wave_read object
    file = wave.open(wav_file_name, 'r')
    samples = file.getnframes()

    duration = samples / rate

    mfcc_feat = mfcc(sig, rate)
    fbank_feat = logfbank(sig, rate)

    print(fbank_feat[1:3,:])


if __name__ == "__main__":
    parser = ArgumentParser(description='Create jpg images and wav file from mp4 input')
    parser.add_argument('--input_file', type=str, default='testvideo1.mp4', help='filepath to read from')
    args = parser.parse_args()

    create_av_files_from_input(args.input_file)
