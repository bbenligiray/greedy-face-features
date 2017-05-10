## Greedy Search for Descriptive Spatial Face Features

My implementation of the following paper:

[Gacav, C.; Benligiray, B.; Topal, C., "Greedy search for descriptive spatial face features," International Conference on Acoustics, Speech and Signal Processing (ICASSP), 2017.](https://arxiv.org/abs/1701.01879)

*Note: The original paper was implemented in C++ by Caner Gacav.
I reproduced the results in python without referring to the original code.
The original code used dlib for SVM, this one uses scikit-learn.*

#### What is this?

Facial landmarks are commonly used for emotion recognition.
However, there are a lot of them, and some are not very descriptive.
This program extracts landmarks from the CK+ dataset, selects the most descriptive ones, and tests this selection.

#### How to run?

Entering the following to the terminal works for a clean install of Ubuntu 16.04.
You will need internet connection to download the required packages.

```
$ cd /path/to/file
$ chmod +x setup.sh
$ ./setup.sh
$ python main.py
```

#### Program Options

You will be asked if you want to extract the features from images, or load the ones I have provided.
If you want to extract the features yourself, you need to download `Emotion_labels.zip` and `extended-cohn-kanade-images.zip` from [the CK+ dataset page](http://www.consortium.ri.cmu.edu/ckagree/), and put them in the project directory.

Then, you will be asked if you want to do the feature selection, or use the example selection result that I have provided.
Feature selection is not a deterministic process, so the final result will not be the same every time.
Overall, the results seem to be about 2% higher than what we have reported in the paper.
