import numpy as np
from sklearn.svm import NuSVC
from sklearn.preprocessing import normalize
from sklearn.metrics import confusion_matrix

import metadata
import data_handler

# finds the best set of features by SFS (aims to maximize mean class accuracy)
def sequential_forward_selection(data, labels):
	no_features = data[0].shape[1]
	
	acc_best = -np.inf
	feats_best = np.zeros(no_features, dtype = bool)

	# select the first two features
	for ind1 in range(no_features):
		for ind2 in range(ind1 + 1, no_features):
			feats_test = np.zeros(no_features, dtype = bool)
			feats_test[ind1] = True
			feats_test[ind2] = True

			acc = test_features(data, labels, feats_test)
		
			if acc > acc_best:
				acc_best = acc
				feats_best = feats_test
	print('Selected features')
	print('Mean class accuracy')
	print(np.argwhere(feats_best)[:, 0])
	print(acc_best)

	while True:
		acc_best_in_search = -np.inf
		feats_best_in_search = np.zeros(no_features, dtype = bool)

		for ind in range(no_features):
			if feats_best[ind]:
				continue
			feats_test = np.copy(feats_best)
			feats_test[ind] = True

			acc = test_features(data, labels, feats_test)

			if acc > acc_best_in_search:
				acc_best_in_search = acc
				feats_best_in_search = feats_test

		if acc_best_in_search > acc_best:
			acc_best = acc_best_in_search
			feats_best = feats_best_in_search
			print(np.argwhere(feats_best)[:, 0])
			print(acc_best)
		else:
			break

	return feats_best

def use_features_for_classification(data, labels, feats_test):
	clf = NuSVC(nu = 0.1, probability = False, decision_function_shape = 'ovo')
	clf.fit(normalize(data[0][:, feats_test]), labels[0])
	preds = clf.predict(normalize(data[1][:, feats_test]))
	return labels[1], preds
	
# trains and tests an SVM
def test_features(data, labels, feats_test):

	labels_true, labels_pred = use_features_for_classification(data, labels, feats_test)
	conf_matrix = confusion_matrix(labels_true, labels_pred)
	conf_matrix = conf_matrix.astype(float)
	mean_class_accuracy = np.mean(np.diagonal(conf_matrix) / np.sum(conf_matrix, axis = 1))

	return mean_class_accuracy
	
def evaluate_final_feature_selection(data_list, labels_list, feats_best):
	# create a generator that segments the dataset in leave-one-subject-out fashion
	gen = data_handler.fold_generator(data_list, labels_list)

	class_overall_tot = np.zeros(metadata.no_classes)
	class_overall_corr = np.zeros(metadata.no_classes)

	labels_true = []
	labels_pred = []
	for ind_subject in range(metadata.no_subjects):
		data, labels = gen.next()
		lt, lp = use_features_for_classification(data, labels, feats_best)
		labels_true.append(lt)
		labels_pred.append(lp)
	labels_true = np.concatenate(labels_true)
	labels_pred = np.concatenate(labels_pred)
	conf_matrix = confusion_matrix(labels_true, labels_pred)
	
	conf_matrix = conf_matrix.astype(float)
	total_accuracy = np.trace(conf_matrix) / np.sum(conf_matrix)
	mean_class_accuracy = np.mean(np.diagonal(conf_matrix) / np.sum(conf_matrix, axis = 1))
	
	print('CK+ labels')
	print(metadata.emotion_labels)
	print('Confusion matrix')
	print(conf_matrix)
	print('Total accuracy')
	print(total_accuracy)
	print('Mean class accuracy')
	print(mean_class_accuracy)
