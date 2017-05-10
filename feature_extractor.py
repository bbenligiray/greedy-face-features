import os
import numpy as np
import skimage.io
import skimage.draw
import dlib

# collects the image paths and labels for fully neutral and fully expressive faces
def itemize_dataset():
	folders_subjects = os.listdir('cohn-kanade-images')
	folders_subjects.sort()
	image_paths = []
	labels = []

	for folder_subjects in folders_subjects:
		image_paths.append([])
		labels.append([])
		folders_emotions = os.listdir(os.path.join('cohn-kanade-images', folder_subjects))
		folders_emotions.sort()
		folders_emotions = [f for f in folders_emotions if not f.startswith('.')]

		for folder_emotions in folders_emotions:

			path_subject_emotion = os.path.join(folder_subjects, folder_emotions)
			
			# skip this emotion if it is not labeled
			if not os.path.exists(os.path.join('Emotion', path_subject_emotion)):
				continue

			files_labels = os.listdir(os.path.join('Emotion', path_subject_emotion))
			
			if not files_labels:
				continue
			# else, add it to the list		

			f = open(os.path.join('Emotion', path_subject_emotion, files_labels[0]), 'r')
			label = f.readline();
			f.close()
			label = label.split()[0]
			label = int(label[0])
			labels[-1].append(label)

			files = os.listdir(os.path.join('cohn-kanade-images', path_subject_emotion))
			files.sort()
			files = [f for f in files if not f.startswith('.')]
			files = [os.path.join('cohn-kanade-images', path_subject_emotion, f) for f in files]
			image_paths[-1].append([files[0], files[-1]])

		# remove this subject if he has no valid labels
		if not labels[-1]:
			del labels[-1]
			del image_paths[-1]

	# this example is mislabeled
	labels[94][0] = 2
	return image_paths, labels

# extracts landmarks, labels, spatial features from CK+
# plots the detected landmarks and dumps them to a folder if plot_landmarks==True
def extract_features(plot_landmarks = False):
	
	# explore the dataset and list its contents
	image_paths, labels = itemize_dataset()

	detector = dlib.get_frontal_face_detector()
	predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

	landmarks = []
	if plot_landmarks:
		if not os.path.exists('landmarks'):
			os.mkdir('landmarks')
	
	# locate landmarks and put them in a list
	for subject in image_paths:
		landmarks.append([])
		for emotion in subject:
			landmarks[-1].append([])
			for image in emotion:
				img = skimage.io.imread(image)
				dets = detector(img, 1)
				det = dets[0]
				shape = predictor(img, det)

				lm = np.zeros([68, 2])
				for ind in range(68):
					lm[ind] = [shape.part(ind).x, shape.part(ind).y]
					if plot_landmarks:
						rr, cc = skimage.draw.circle(lm[ind][1], lm[ind][0], 3)
						img[rr, cc] = 255
						rr, cc = skimage.draw.circle(lm[ind][1], lm[ind][0], 2)
						img[rr, cc] = 0
				landmarks[-1][-1].append(lm)
				if plot_landmarks:
					skimage.io.imsave(os.path.join('landmarks', os.path.split(image)[-1]), img)

	# calculate spatial features and put them in a list
	spatial_features = []
	for subject in landmarks:
		spatial_features.append([])
		for emotion in subject:
			no_landmarks = emotion[0].shape[0]
			dist_for_two_images = []
			for image in emotion:
				dist_curr_image = np.zeros([no_landmarks * (no_landmarks - 1)])
				counter = 0
				for ind_lm1 in range(no_landmarks):
					for ind_lm2 in range(ind_lm1 + 1, no_landmarks):
						dist_curr_image[counter] = image[ind_lm1, 0] - image[ind_lm2, 0]
						counter += 1
						dist_curr_image[counter] = image[ind_lm1, 1] - image[ind_lm2, 1]
						counter += 1
				dist_for_two_images.append(dist_curr_image)
			spatial_features[-1].append(dist_for_two_images[0] - dist_for_two_images[1])

	return landmarks, labels, spatial_features
