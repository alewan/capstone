from argparse import ArgumentParser
import os
import subprocess
from python_speech_features import mfcc, logfbank
import scipy.io.wavfile as wav
from scipy import signal
import matplotlib.pyplot as plt
import numpy as np
import wave
import librosa 
import librosa.display

def create_av_files_from_input(file_path):

    if not os.path.isfile(file_path):
        print("File DNE")
        return "File doesn't exist" 

    #Raw name without mp4 extension
    file_name = os.path.split(file_path)
    name = os.path.splitext(file_name[1])[0]

    #Arbitrarily set path names, one contains garbage,other contains final nn images 
    #SET THESE
    trash_path = "../pipeline/pipeline_helper/trash/"
    keep_image_path = "../pipeline/pipeline_helper/images_preprocessed/"
    keep_audio_path = "../pipeline/pipeline_helper/audio_preprocessed/"

    #names 
    jpg_output_name = keep_image_path + name  
    aud_output_name = trash_path + "split"
    wav_file_name = trash_path + name + ".wav"
    silrem = trash_path + "sil_rem" + name + ".mp4" 
    rgb_path = keep_image_path + name + "_rgb_plt"
    rgb_path_audio = keep_audio_path + name + "_rgb_plt"
    #rgb_path = "/Users/meghanmuldoon/Desktop/rgbplt"

    #Strip audio and combine audio channels 
    command1 = "(ffmpeg -i " + file_path + " -vn -ac 1 " + wav_file_name + ")"
    subprocess.call(command1, shell=True)

    (rate, sig) = wav.read(wav_file_name)
    sig = sig.astype(np.float32)

    #Plot of original signal
    # plt.plot(sig, linewidth=0.5)
    # plt.savefig( trash_path + "full_wave.png")
    # plt.close()

    #Take in the first 2000 samples of the signal 
    silence = sig[:50000]
    maximum = np.amax(np.abs(silence))

    first_max_index = np.argmax(np.abs(sig) > maximum)
    last_max_index = np.size(sig) - np.argmax(np.abs(np.flip(sig)) > maximum)

    #If the video accidently only contains digital silence
    if(sig.any() != 0):

        #subtract 2000 frames to be safe and more at end (resonance)
        first_max_index = first_max_index - 2000
        last_max_index = last_max_index + 6000

        new_sig = sig[first_max_index: last_max_index]

        sig_start_time = str(first_max_index / rate)
        sig_end_time = str(last_max_index / rate)

  		#Plot of 
        # plt.ylim(-3000, 3000)
        # plt.plot(new_sig, linewidth=0.5)
        # plt.savefig( trash_path+"sil_rem.png")
        # plt.close()
        
        #Remove the silence 
        command2 = "(ffmpeg -i " + file_path + " -ss "+ sig_start_time + " -t " + sig_end_time + " "+ silrem+ ")"
        subprocess.call(command2, shell=True)

        # COMMAND TO SAMPLE THE FAMES
        # Sample = 5/3 fps (Sample every 0.666... seconds)
        command3 = "(ffmpeg -i " + silrem + " -vf fps=5/3 " + jpg_output_name + "_%04d.jpg -hide_banner) "
        subprocess.call(command3, shell=True)

        #Split silence removed into 0.666.. second segments 
        command4 = "(ffmpeg -i " + wav_file_name + " -f segment -segment_time 0.6666 " + aud_output_name + "_%04d.wav -hide_banner) "
        subprocess.call(command4, shell=True)
    
    #Spectral processing 
    #Remove everything except for splits from trash 
    os.remove(wav_file_name)
    os.remove(silrem)

    files = os.listdir(trash_path)
    files_no_hidden = []

    for filename in files:
    	if not filename.startswith('.'):
    		files_no_hidden.append(filename)
    i = 1		
    for filename in files_no_hidden:
    	file_path = trash_path + "/" +filename
    	(rate, sig) = wav.read(file_path)
    	sig = sig.astype(np.float32)
    	file = wave.open(file_path, 'r')
    	samples = file.getnframes()
    	duration = samples / rate
    	
    	if duration > 0.66 :
    		MFCC = librosa.feature.melspectrogram(sig, rate, hop_length = 256, window = np.hamming, n_mels = 200)
    		MFCC_DB = librosa.power_to_db(MFCC, ref=np.max)
    		delta_mfcc = librosa.feature.delta(MFCC_DB, width = 7, order = 1)
    		D_MFCC_DB = delta_mfcc
    		delta_delta_mfcc = librosa.feature.delta(MFCC_DB, width = 7, order = 2)
    		D_D_MFCC_DB = delta_delta_mfcc

    		mfcc_min = -80 #np.min(MFCC_DB)
    		mfcc_max = 0 #np.max(MFCC_DB)
    		mfcc_feat = 255 * ((MFCC_DB - mfcc_min) / (mfcc_max - mfcc_min))
    		mfcc_feat = np.clip(mfcc_feat, 0, 255).astype(np.uint8)
    		
    		delta_mfcc_min = -10 #np.min(D_MFCC_DB)
    		delta_mfcc_max = 10 #np.max(D_MFCC_DB)
    		delta_mfcc_feat = 255 * ((D_MFCC_DB - delta_mfcc_min)/ (delta_mfcc_max - delta_mfcc_min))
    		delta_mfcc_feat = np.clip(delta_mfcc_feat, 0, 255).astype(np.uint8)

    		d_delta_mfcc_min = -10#np.min(D_D_MFCC_DB)
    		d_delta_mfcc_max = 10 #np.max(D_D_MFCC_DB)
    		d_delta_mfcc_feat = 255 * ((D_D_MFCC_DB - d_delta_mfcc_min)/ (d_delta_mfcc_max - d_delta_mfcc_min))
    		d_delta_mfcc_feat = np.clip(d_delta_mfcc_feat, 0, 255).astype(np.uint8)

    		#Stack the three arrays 
    		rgb = np.dstack((mfcc_feat, delta_mfcc_feat, d_delta_mfcc_feat))

    		#Plot the RGB Array
    		fig, ax = plt.subplots()
    		mfcc_data= np.swapaxes(rgb, 0 ,1)
    		cax = ax.imshow(rgb, interpolation='nearest', origin='lower')
    		ax.set_title('rgb')
    		plt.imsave(rgb_path_audio+"_000"+str(i)+ ".png", rgb)
    		plt.close()
    		i = i + 1

    #Remove everything in the garbage folder 
    files = os.listdir(trash_path)
    for filename in files:
    	file_path = os.path.join(trash_path, filename)
    	os.remove(file_path)

if __name__ == "__main__":
	input_file_path = "../pipeline_helper/user_upload/input_file.mp4"
	create_av_files_from_input(input_file_path)
