# **Behavioral Cloning** 

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/center_2016_12_01_13_30_48_404.png "center image"
[image2]: ./examples/center_2016_12_01_13_30_48_404_flip.png "flipped image"
[image3]: ./examples/center_2016_12_01_13_30_48_404_crop.png "cropped image"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* video.mp4 recorded video in autonomous mode
* writeup_report.md summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, through commented and self-explaining code.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

My model is based on the NVIDIA network for autonomous driving. It uses RELU activations and dropout for hidden layers.

#### 2. Attempts to reduce overfitting in the model

The model contains dropout layers in order to reduce overfitting (model.py lines 76-89).
I used different dropout rates for convolutional (smaller) and fully connected layer, as per most recent best practices.
Being the last hidden layer quite small, I prefered to use a little L2 regularizer instead of dropout.

The model was trained and validated on different data sets to ensure that the model was not overfitting.
In particular, I augmented the training set adding the side camera images, associated with a certain correction at the steering wheel, in order to provide information in case of critical situation. 
I also flipped the center image and its associated steering angle, in order to eliminate bias in the training set. I decided not to do the same for side images, as they are just meant for more extreme corrections.

The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.
In order to complete track1, training on track1 only was sufficient.
It would not be sufficient for track2, where uphill and downhill situations as well as very tight curves are present.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 96).
I also used a 0.1 drop out rate for convolutional layers, and a 0.33 drop out rate for fully connected ones.
For the last hidden layer as well as the output layer I used a 0.001 L2 regularizer.

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. 
As already mentioned, I used a combination of center lane driving, recovering from the left and right sides of the road and flipping the center image.

For details about how I created the training data, see the next section. 

### Model Architecture and Training Strategy

#### 1. Model Architecture

My first choice, the NVIDIA architecture, was successful at first attempt. The layers detail follows:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 160x320x3 RGB image							| 
| Lambda         		| Normalize between -0.5 and 0.5				|
| Cropping         		| 60 pixels on the top, 20 on the bottom		| 
| Convolution 5x5     	| 2x2 stride, 24 channels					 	|
| RELU					|												|
| Dropout				| drop rate 0.1									|
| Convolution 5x5	    | 2x2 stride, 36 channels						|
| RELU					|												|
| Dropout				| drop rate 0.1									|
| Convolution 5x5	    | 2x2 stride, 48 channels					 	|
| RELU					|												|
| Dropout				| drop rate 0.1									|
| Convolution 3x3	    | 64 channels								 	|
| RELU					|												|
| Dropout				| drop rate 0.1									|
| Convolution 3x3	    | 64 channels								 	|
| RELU					|												|
| Dropout				| drop rate 0.1									|
| Flatten				| 			 									|
| Fully connected		| outputs 100 									|
| RELU					|												|
| Dropout				| drop rate 0.33								|
| Fully connected		| outputs 50 									|
| RELU					|												|
| Dropout				| drop rate 0.33								|
| Fully connected		| outputs 10, L2 regularizer 0.001				|
| RELU					|												|
| Fully connected		| outputs 1 , L2 regularizer 0.001				|

#### 3. Creation of the Training Set & Training Process

Even if I did record three laps in training mode, the training data provided were already sufficient to successfully and accurately complete the assignment

As already mentioned, to augment the data set, I also flipped images and angles in order to eliminate training set bias. For example, here is an image that has then been flipped:

![alt text][image1]
![alt text][image2]

The model pipeline has also an image cropping layer, in order to focus on the section of the image that really matters.

![alt text][image3]

I also used side cameras, adding a correction factor of +/-0.2 to the steering angle. This would help to control the car in case it approximates too much the side of the road.

As evident from the model description, I also introduced a normalization Lambda layer, for a better numerical conditioning.

I finally randomly shuffled the data set and put 20% of the center image data into a validation set. Observe that my augmentation process multiplied by 4 the training set.

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. 
After 3 epochs, the training loss tended to saturate, for this reason I din not increase this number.
I used an adam optimizer so that manually training the learning rate wasn't necessary.
