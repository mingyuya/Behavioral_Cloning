# Writeup for P3:Behavioral Cloning

### Matt, Min-gyu, Kim
---

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/placeholder.png "Model Visualization"
[image2]: ./examples/placeholder.png "Grayscaling"
[image3]: ./examples/placeholder_small.png "Recovery Image"
[image4]: ./examples/placeholder_small.png "Recovery Image"
[image5]: ./examples/placeholder_small.png "Recovery Image"
[image6]: ./examples/placeholder_small.png "Normal Image"
[image7]: ./examples/placeholder_small.png "Flipped Image"

## Rubric Points
#### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Required Files

Here is the list of the files in my submission:

| File | Description |
| ------ | ----- |
| model.py | Containing the script to define and train the model |
| drive.py | For driving the car in autonomous mode |
| model.h5 | Containing a trained convolution neural network |
| README.md | Writeup for summarizing the results |
| video.mp4 | The video clip which is the result of simulation |

### Quality of Code
1. ```model.py``` has the pipeline for training, validating and saving the model. For the training stage, Python generator is used because of the model based on Nvidia's one which has so many parameter (Line: 39-66). It can prevent lack of memories during the training.

2. The network architecture is drawn at the end of ```model.py``` (Line: 135-136).

3. Using the provided simulator and drive.py file, the car can be driven autonomously and safely around the track by executing ```python drive.py model.h5```

### Model Architecture and Training Strategy

#### 1. Structure
 My model has been built based on the [Nvidia's model](https://devblogs.nvidia.com/parallelforall/deep-learning-self-driving-cars/) which is consisted of single normalization layer at the input stage, 5x5 convolutional layers and 3 fully connected layers. [ELU (Exponential Linear Unit)](https://arxiv.org/abs/1511.07289) is used as the activation layers and [Batch Normalization](http://cs231n.github.io/neural-networks-2/#batchnorm) layers are placed in before the each of activation layers for fast and accurate optimization. The detail is described in the following table.

#### 2. To avoid overfittig
 Dropout layer with the 0.8 of is for avoiding the model is overfitted to the data correspond to move straight forward. If Dropout layer has the rate higher than 0.8 or is not exist, the model after training pushed the car to outside when it is passing curve

| Layer | Description | model.py lines |
| ------ | ----- | :-----: |
| Lambda | Normalize the input value between -1 and +1 | 85 |
| Cropping2D | For driving the car in autonomous mode | 86 |
| Conv2D | 5x5x24, strides=(2,2), padding=valid | 87 |
| BatchNormalization | | 88 |
| Activation | Exponential Linear Unit | 89 |
| Conv2D | 5x5x36, strides=(2,2), padding=valid | 90 |
| BatchNormalization | | 91 |
| Activation | Exponential Linear Unit | 92 |
| Conv2D | 5x5x48, strides=(2,2), padding=valid | 93 |
| BatchNormalization | | 94 |
| Activation | Exponential Linear Unit | 95 |
| Conv2D | 3x3x64, strides=(1,1), padding=valid | 96 |
| BatchNormalization | | 97 |
| Activation | elu | 98 |
| Conv2D | 3x3x64 | 99 |
| BatchNormalization | | 100|
| Activation | elu | 101 |
| Flatten | | 102 |
| Dropout | Rate = 0.8 | 103 |
| Fully Connected | 1164 fan-in, 100 fan-out | 104 |
| BatchNormalization | | 105 |
| Activation | elu | 106 |
| Fully Connected | 50 fan-out| 107 |
| BatchNormalization | | 108 |
| Activation | elu | 109 |
| Fully Connected | 10 fan-out | 110 |
| BatchNormalization| | |
| Fully Connected | 1 fan-out | |

#### 3. Opimizer and Training Parameter

The model used an **adam optimizer**. The learning rate was setted to **0.01** and was not tunned. Beacause Batch Normalization layer solves Vanishing-Gradient and Bad-Initial-Value Problem  

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road ... 

For details about how I created the training data, see the next section. 

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to ...

My first step was to use a convolution neural network model similar to the ... I thought this model might be appropriate because ...

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that my first model had a low mean squared error on the training set but a high mean squared error on the validation set. This implied that the model was overfitting. 

To combat the overfitting, I modified the model so that ...

Then I ... 

The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track... to improve the driving behavior in these cases, I ....

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

#### 2. Final Model Architecture

The final model architecture (model.py lines 18-24) consisted of a convolution neural network with the following layers and layer sizes ...

Here is a visualization of the architecture (note: visualizing the architecture is optional according to the project rubric)

![alt text][image1]

#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using center lane driving. Here is an example image of center lane driving:

![alt text][image2]

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to .... These images show what a recovery looks like starting from ... :

![alt text][image3]
![alt text][image4]
![alt text][image5]

Then I repeated this process on track two in order to get more data points.

To augment the data sat, I also flipped images and angles thinking that this would ... For example, here is an image that has then been flipped:

![alt text][image6]
![alt text][image7]

Etc ....

After the collection process, I had X number of data points. I then preprocessed this data by ...


I finally randomly shuffled the data set and put Y% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was Z as evidenced by ... I used an adam optimizer so that manually training the learning rate wasn't necessary.
