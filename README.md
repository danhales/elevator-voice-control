# Hey, Elevator

The goal of this project is to create a speech recognition system that can be deployed in a microcontroller to enable hands-free operation (voice control) of elevators. The intended purpose is to reduce physical interactions with high-contact surfaces that may <a href="https://www.businessinsider.com/coronavirus-jumped-between-people-via-elevator-surfaces-study-2020-7">contribute to the spread of COVID-19</a>. Although the primary mode of transmission is known to be <a href="https://www.cdc.gov/coronavirus/2019-ncov/prevent-getting-sick/how-covid-spreads.html?CDC_AA_refVal=https%3A%2F%2Fwww.cdc.gov%2Fcoronavirus%2F2019-ncov%2Fprepare%2Ftransmission.html">airborne droplets</a>, eliminating as many potential routes of transmission as possible will help to slow the spread.

This project was completed as my capstone at Flatiron School DC's <a href="https://flatironschool.com/career-courses/data-science-bootcamp/dc">Data Science</a> program.

## Data
In this project, I train a series of neural networks on Google's <a href="https://ai.googleblog.com/2017/08/launching-speech-commands-dataset.html">Speech Commands Dataset</a>. In particular, we focused on the words "zero", "one", "two", "three", "four", "five", "six", "seven", "eight", and "nine", because these can be combined to form floor numbers in elevators. In particular, I worked only with the files that contained exactly one second of audio to facilitate processing and avoid unpredictable padding.

To process these files, I downsampled each to 8000 Hz (which captures the majority of audio data in <a href="https://en.wikipedia.org/wiki/Voice_frequency">normal human speech</a>), then compared processing techniques across the same neural network architecture.

This dataset is too large to be hosted in this repo, so it can either be downloaded from the source above or from the first non-markdown cell in <b>mvp.ipynb</b>.

## What's in this repository?

As of 9/1/20, this repository contains the following files:

### <b>mvp.ipynb</b> 
This is the Jupyter notebook for my minimum viable product (MVP). Each cell in this notebook is self-contained, to show each step of the development process. A brief summary of the code you will find in it:

* Pre-processing steps, including organizing the metadata so files could be loaded. Several different functions are provided to load the data in different formats (raw (sampling rate: 8000), a modified STFT (mean of absolute value across time windows), flattened MFCCs, and a unflattened MFCCs.

* Training the same, very simple baseline model (a single hidden layer of 256 fully-connected units) to compare performance of the neural network when trained on the raw data, the modified STFTs, and the flattened MFCCs.

* A simple CNN which consists of a single convolutional layer (followed by a flattening layer) before the `Dense` layer in the baseline model.

* Analysis of the validation accuracy (10% of the training data, or 8% of the overall data) of each of these models when fitted with the same parameters. The model is allowed to run for longer than necessary in order to examine the tendency for overfitting. Although validation accuracy is not the best metric for evaluating a final model, it is a good heuristic for pointing us in the right direction.

* A brief discussion of how this model could be deployed, effects and limitations of the model, and some ethical issues involved in speech recognition systems.

### <b>data_loader.py</b>
For your convenience, all output should already be displayed in the notebook so you don't have to run the code yourself. If you want to run anything, the functions in the notebook have been stored in <b>data_loader.py</b>, so all cells should be entirely self-contained.

### <b>requirements.txt</b>
The conda-generated requirements file for the environment I used in developing this MVP. You can <a href="https://www.idkrtm.com/what-is-the-python-requirements-txt/">use this file</a> for building an equivalent environment for running this model.

### <b>metadata.csv</b> 
This is a file produced by a cell in the mvp notebook. It contains four columns. `filename` is the filepath from this notebook's directory through the subdirectory of `speech_commands_v0.01`, into the digit's subdirectory, to each individual recording. `rec_id` is a unique identifier, computed by hashing the filename, and is used as an index in processing. `digit` contains the digit being spoken, and is equivalent to the name of the directory in which it is found. `length` is the number of samples in the file when loaded with a sampling rate of 8000.

### <b>history</b>
A directory containing the pickled history of our four models, so we can quickly compare them at the end without eating up system resources.

## Results

As can be expected, the raw data performed the worst (18% validation accuracy). The STFT data gave us a large boost in validation accuracy (up to 67%), and using the flattened MFCCs pushed the validation accuracy to around 83%. The convolutional layer raised the accuracy to about 88%. As can be expected, all models saw a large degree of overfitting, with the most egregious being the raw data (especially when the model was allowed to run for 50 epochs). 

This tells us that the transformation from raw data to MFCC is absolutely a step in the right direction, but more investigation is necessary to determine exactly the right architecture.

## Deployment

This model could be deployed in a microcontroller, such as a Raspberry Pi, which has been modified to interface with the elevator controller. Although there are existing speech recognition models, this model is open-source and entirely self-contained (once it has been trained).

## Next Steps

Now that the MVP is completed, here are my plans:

* Investigate the confusion matrix for the CNN to find out where the model can be improved. In order to reduce as much contact as possible with the elevator surfaces, it makes sense to optimize the model for the most-used elevator buttons––I'm interpreting this as the "ground floor," or "one."

* Adding in additional hidden layers to the flattened MFCC model to see if the increase in accuracy in the CNN was due to the added <i>convolution</i>, or  the added <i>depth</i>.

* Introducing regularization parameters to decrease overfitting to the training data.

* Deeper data cleaning. Using noise floor detection to remove background sound, and discarding dead space in the data. Although all of the recordings were of uniform length, the CNN may be more useful if the important speech features start exactly at the beginning of the clip, instead of at a variety of positions. The ending could be padded with zeros for uniform length.

* Get more data and increase the noise in the dataset

* Deployment in an interactive manner via a web app which allows the speaker to record themselves speaking a digit, feed it into the neural network, and have it classified.
