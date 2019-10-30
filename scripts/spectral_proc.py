from argparse import ArgumentParser
import os
import subprocess
from python_speech_features import mfcc, logfbank, delta
import librosa 
import librosa.display
import scipy.io.wavfile as wav
from scipy import signal
import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
import wave

def spectral_analysis(path: str, out_path: str):


	if not os.path.isdir(path):
		print("Dir DNE")
		return "Dir doesn't exist"

	#Files = a list of filenames in the current directory
	files = os.listdir(path)	
	
	#Manually remove hidden files (.DS_Store) from the filelist
	#Don't use glob to remove hidden files
	
	files_no_hidden = []
	for filename in files:
		if not filename.startswith('.'):
			files_no_hidden.append(filename)
	i = 0
	for filename in files_no_hidden:

		#Specify file paths, mostly unneeded 
		file_path = path + "/" +filename

		image_path = out_path + "/" + filename + "mfcc_plt"
		sig_path1 = out_path + "/" + filename + "sig1_plt"
		sig_path2 = out_path + "/" + filename + "sig2_plt"
		spec_path = out_path + "/" + filename + "spec_plt"
		spec2_path = out_path + "/" + filename + "spec_emph_plt"
		delt_path = out_path + "/" + filename + "delt_plt"
		d_delt_path = out_path + "/" + filename + "d_delt_plt"
		
		rgb_path = out_path + "/" + filename+ "rgb_plt"


		#Sample the single path wav file and plot regular spectrum
		(rate, sig) = wav.read(file_path)
		sig = sig.astype(np.float32)

		file = wave.open(file_path, 'r')
		samples = file.getnframes()
		duration = samples / rate

		print(duration)

		#Ignore small clips at the end of the seg data
		if duration < 0.5 :
			return

		frequencies, times, spectrogram = signal.stft(sig, rate)
		plt.pcolormesh(times, frequencies, np.abs(spectrogram))
		plt.savefig(spec_path + ".png")
		plt.close()


		#Plot the signals themselves (emphasized and not emphasized)
		plt.ylim(-3500, 3500)
		plt.plot(sig, linewidth=0.5)
		plt.savefig(sig_path1 + ".png")
		plt.close()


		#hop_length = 0.025 seconds
		#Compute MFCC with librosa and convert to decibles 
		MFCC = librosa.feature.melspectrogram(sig, rate, hop_length = 256, window = np.hamming, n_mels = 200)
		MFCC_DB = librosa.power_to_db(MFCC, ref=np.max)

		#Find diferentials of MFCC
		delta_mfcc = librosa.feature.delta(MFCC_DB, width = 7, order = 1)
		D_MFCC_DB = delta_mfcc

		delta_delta_mfcc = librosa.feature.delta(MFCC_DB, width = 7, order = 2)
		D_D_MFCC_DB = delta_delta_mfcc

		#normalize the values between 0 and 255 to put elements into colour channels 
		#MFCC_DB: max = 0, min = -80
		mfcc_min = -80 #np.min(MFCC_DB)
		mfcc_max = 0 #np.max(MFCC_DB)
		mfcc_feat = 255 * ((MFCC_DB - mfcc_min) / (mfcc_max - mfcc_min))
		mfcc_feat = np.clip(mfcc_feat, 0, 255).astype(np.uint8)

		delta_mfcc_min = -10 #np.min(D_MFCC_DB)
		delta_mfcc_max = 10 #np.max(D_MFCC_DB)
		print("delta_mfcc_db min: ", delta_mfcc_min, "delta_mfcc_db max:", delta_mfcc_max)
		delta_mfcc_feat = 255 * ((D_MFCC_DB - delta_mfcc_min)/ (delta_mfcc_max - delta_mfcc_min))
		delta_mfcc_feat = np.clip(delta_mfcc_feat, 0, 255).astype(np.uint8)

		d_delta_mfcc_min = -10#np.min(D_D_MFCC_DB)
		d_delta_mfcc_max = 10 #np.max(D_D_MFCC_DB)
		print("d_delta_mfcc_db min: ", d_delta_mfcc_min, "d_delta_mfcc_db max:", d_delta_mfcc_max)
		d_delta_mfcc_feat = 255 * ((D_D_MFCC_DB - d_delta_mfcc_min)/ (d_delta_mfcc_max - d_delta_mfcc_min))
		d_delta_mfcc_feat = np.clip(d_delta_mfcc_feat, 0, 255).astype(np.uint8)

		#Stack the three arrays 
		rgb = np.dstack((mfcc_feat, delta_mfcc_feat, d_delta_mfcc_feat))

		#Plot the RGB Array
		fig, ax = plt.subplots()
		mfcc_data= np.swapaxes(rgb, 0 ,1)
		cax = ax.imshow(rgb, interpolation='nearest', origin='lower')
		ax.set_title('rgb')
		plt.imsave(rgb_path + ".png", rgb)
		plt.close()

		#Plot the MFCC, delta and delta delta for fun
		fig, ax = plt.subplots()
		mfcc_data= np.swapaxes(D_MFCC_DB, 0 ,1)
		cax = ax.imshow(D_MFCC_DB, interpolation='nearest', origin='lower')
		ax.set_title('delta')
		plt.savefig(delt_path + ".png")
		plt.close()

		fig, ax = plt.subplots()
		mfcc_data= np.swapaxes(MFCC_DB, 0 ,1)
		cax = ax.imshow(MFCC_DB, interpolation='nearest', origin='lower')
		ax.set_title('MFCC')
		np
		plt.savefig(image_path + ".png")
		plt.close()

		fig, ax = plt.subplots()
		mfcc_data= np.swapaxes(D_D_MFCC_DB, 0 ,1)
		cax = ax.imshow(D_D_MFCC_DB, interpolation='nearest', origin='lower')
		ax.set_title('DELTA delta')
		plt.savefig(d_delt_path + ".png")
		plt.close()
		
		'''
		# Simple librosa display of MFCC
		librosa.display.specshow(MFCC_DB, x_axis='time', y_axis='mel');
		plt.title("")
		plt.savefig(rgb_path + ".png")
		plt.close()

		'''
		#print(spectrogram) 
		

if __name__ == "__main__":
	parser = ArgumentParser(description='analyze a given wav file')
	parser.add_argument('--input_dir', type=str, default='testvideo1.mp4', help='filepath to wav files')
	parser.add_argument('--output_folder', type=str, default='~/Desktop', help='desired output location')
	args = parser.parse_args()

	spectral_analysis(args.input_dir, args.output_folder)
