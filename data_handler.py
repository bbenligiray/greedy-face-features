import numpy as np

# converts the dataset list into a numpy array, discarding the subject data in the process
def convert_dataset_list_to_numpy_array(data, labels):
	
	no_subjects = len(data)
	no_examples = 0
	for ind_subject in range(no_subjects):
		no_examples += len(data[ind_subject])
	no_features = data[0][0].shape[0]
	
	data_out = np.ndarray([no_examples, no_features])
	labels_out = np.ndarray(no_examples)

	counter = 0
	for ind_subject in range(no_subjects):
			for ind_emotion in range(len(data[ind_subject])):
				data_out[counter] = data[ind_subject][ind_emotion]
				labels_out[counter] = labels[ind_subject][ind_emotion]
				counter += 1

	return data_out, labels_out

# gets the dataset in numpy format as input, segments it into two
# protects the class frequencies
def segment_dataset(data, labels, val_ratio):

	inds_sorted = labels.argsort()
	labels = labels[inds_sorted]
	data = data[inds_sorted]

	inds_val = np.zeros(len(data), dtype = bool)
	no_class = int(max(labels))

	for ind_class in range(1, no_class + 1):
		inds_class = labels == ind_class
		no_example_by_class = sum(inds_class)
		no_val_by_class = int(round(sum(inds_class) * val_ratio))
		ind_class_start = np.argmax(labels == ind_class)
		selected_val = np.random.choice(range(ind_class_start, ind_class_start + no_example_by_class), no_val_by_class)
		inds_val[selected_val] = True

	data_val = data[inds_val]
	labels_val = labels[inds_val]
	
	data_train = data[np.invert(inds_val)]
	labels_train = labels[np.invert(inds_val)]

	data_train, labels_train = shuffle_arrays(data_train, labels_train)
	data_val, labels_val = shuffle_arrays(data_val, labels_val)

	return [data_train, data_val], [labels_train, labels_val]
	
# generates folds in leave-one-subject-out fashion
def fold_generator(data, labels):
	no_subjects = len(data)

	for ind_out in range(no_subjects):

		data_test = np.array(data[ind_out])
		labels_test = np.array(labels[ind_out])

		data_copy = list(data)
		del data_copy[ind_out]
		labels_copy = list(labels)
		del labels_copy[ind_out]

		data_train, labels_train = convert_dataset_list_to_numpy_array(data_copy, labels_copy)
		data_train, labels_train = shuffle_arrays(data_train, labels_train)

		yield [data_train, data_test], [labels_train, labels_test]
	
def shuffle_arrays(arr1, arr2):
	assert len(arr1) == len(arr2)
	p = np.random.permutation(len(arr1))
	return arr1[p], arr2[p]
