## Greedy Search for Descriptive Spatial Face Features

My implementation of the following paper:

[Gacav, C.; Benligiray, B.; Topal, C., "Greedy search for descriptive spatial face features," International Conference on Acoustics, Speech and Signal Processing (ICASSP), 2017.](https://arxiv.org/abs/1701.01879)

The original paper was implemented by Caner Gacav in C++.
I reproduced the results in python without referring to the original code.
The original code used dlib for SVM, this one uses scikit-learn.
The results seem to be about 2-4% higher than what we have reported in the paper, probably because of the difference in SVM parameters/implementation.

#### What is this?

* Spatial features are derived from displacements of facial landmarks. They are a kind of geometric feature that can be used for facial expression recognition.
* A large number of spatial features can be extracted from a face, but they are not all equally descriptive.
* In the face expression recognition literature, geometric features are hand-picked, dimension-reduced or used as is, with the redundancy.
* In this study, we use sequential forward selection to obtain a small subset of spatial features that describes the facial expressions well.
* In the figure below, you see an example subset. The changes in the indicated vertical or horizontal distances are the selected spatial features.
* The proposed method delivers 88.7% recognition accuracy in the CK+ dataset, which is the highest performance among the methods that only use geometric features.

<p align="center">
  <img src="https://cloud.githubusercontent.com/assets/19530665/26025346/2287308a-37ee-11e7-84c0-311a67de3465.png"/>
</p>

#### How to run?

Entering the following to the terminal works for a clean install of Ubuntu 16.04.
You will need Internet connection to download the required packages.

```
$ cd /path/to/file
$ chmod +x setup.sh
$ ./setup.sh
$ python main.py
```

#### Program Options

You will be asked if you want to extract the features from images, or load the ones I have provided.
If you want to extract the features yourself, you need to download `Emotion_labels.zip` and `extended-cohn-kanade-images.zip` from [the CK+ dataset page](http://www.consortium.ri.cmu.edu/ckagree/), and put them in the project directory.

Then, you will be asked if you want to run the feature selection, or use the provided example selection.
Feature selection is not a deterministic process, so the final result will not be the same every time.
