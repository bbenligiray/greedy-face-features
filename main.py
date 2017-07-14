import shutil
import os
import zipfile
import urllib
import bz2
import pickle
import numpy as np

import data_handler
import feature_extractor
import feature_selector

global no_class
no_class = 7

# loads spatial features and labels from CK+ images into a list
# each element of this list is a subject
# each subject has a list of emotions
# each emotion has a spatial feature vector of length 4556
def load_features():
	with open('ck+preprocessed.p', 'rb') as f:
		d = pickle.load(f)
	return d['spatial_features'], d['labels']

def write_features(landmarks, labels, spatial_features):
	d = {'landmarks': landmarks, 'labels': labels, 'spatial_features': spatial_features}
	with open('ck+preprocessed.p', 'wb') as f:
		pickle.dump(d, f, protocol = pickle.HIGHEST_PROTOCOL)

def main():

	extract_features = False
	plot_landmarks = False
	select_features = False
	
	print('Greedy Search for Descriptive Spatial Face Features, ICASSP 2017')
	
	print('Enter the number to choose:')
	print('1 - Extract spatial features from images (needs CK+ dataset)')
	print('2 - Load previously extracted spatial features')
	
	while True:
		choice = raw_input()
		if choice == '1':
			extract_features = True
			print('Plot the landmarks and dump them in a directory? (y/n)')
			while True:
				choice = raw_input().lower()
				if choice == 'y':
					plot_landmarks = True
					break
				elif choice == 'n':
					plot_landmarks = False
					break
			break
		elif choice == '2':
			extract_features = False
			break
				
	print('Enter the number to choose:')
	print('1 - Do the feature selection (may take several hours)')
	print('2 - Test previously selected spatial features')
	
	while True:
		choice = raw_input()
		if choice == '1':
			select_features = True
			break
		elif choice == '2':
			select_features = False
			break
	
	if extract_features:
		# check if we have the .zip files
		if not os.path.isfile('Emotion_labels.zip'):
			raise FileNotFoundError('Emotion_labels.zip not found. Download them from CK+ dataset and put in greedy-face-features folder.')
		if not os.path.isfile('extended-cohn-kanade-images.zip'):
			raise FileNotFoundError('extended-cohn-kanade-images.zip not found. Download them from CK+ dataset and put in greedy-face-features folder.')
			
		# clear files remaining from the previous run
		shutil.rmtree('cohn-kanade-images', ignore_errors = True)
		shutil.rmtree('Emotion', ignore_errors = True)
		shutil.rmtree('landmarks', ignore_errors = True)
			
		# unzip the dataset to the project directory
		zip_ref = zipfile.ZipFile('extended-cohn-kanade-images.zip', 'r')
		zip_ref.extractall()
		zip_ref.close()
		zip_ref = zipfile.ZipFile('Emotion_labels.zip', 'r')
		zip_ref.extractall()
		zip_ref.close()
		
		# download and extract the dlib file required for landmark localization
		if not os.path.isfile('shape_predictor_68_face_landmarks.dat'):
			urllib.urlretrieve('http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2', 'shape_predictor_68_face_landmarks.dat.bz2')
			decompressedData = bz2.BZ2File('shape_predictor_68_face_landmarks.dat.bz2').read()
			f = open('shape_predictor_68_face_landmarks.dat', 'wb')
			f.write(decompressedData)
			f.close()
			os.remove('shape_predictor_68_face_landmarks.dat.bz2')
	
		# extract features and labels (we do not use the landmarks, but keep them anyway)
		landmarks, labels, spatial_features = feature_extractor.extract_features(plot_landmarks = plot_landmarks)
		write_features(landmarks, labels, spatial_features)
		data_list = spatial_features
		labels_list = labels
		
	else:
		if not os.path.isfile('ck+preprocessed.p'):
			raise FileNotFoundError('ck+preprocessed.p not found. Need to extract landmarks.')
		data_list, labels_list = load_features()
	

	# data_list, labels_list keep the dataset as a list of lists
	# the upper list is a list of subjects, the lower lists are lists of emotions of each subject
	# we discard the subject information and put the features in a single numpy array
	data_np, labels_np = data_handler.convert_dataset_list_to_numpy_array(data_list, labels_list)
	
	if select_features:
		# segment the dataset: 0.6 training / 0.4 test
		data_seg, labels_seg = data_handler.segment_dataset(data_np, labels_np, 0.4)
		feats_best = feature_selector.sequential_forward_selection(data_seg, labels_seg)
	else:
		feats_best = np.zeros(4556, dtype = bool)
		feats_best[356] = True
		feats_best[429] = True
		feats_best[1099] = True
		feats_best[2559] = True
		feats_best[3083] = True
		feats_best[3862] = True
		feats_best[4080] = True
		feats_best[4277] = True
		
	# evaluates the selection and prints the results
	feature_selector.evaluate_final_feature_selection(data_list, labels_list, feats_best)

if __name__ == '__main__':
	main()