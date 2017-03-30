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

[image1]: ./figures/angle_org.png "Data Distribution"
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
 
1) **I started from [Nvidia's model](https://devblogs.nvidia.com/parallelforall/deep-learning-self-driving-cars/) **. 
2) **[Batch Normalization](https://arxiv.org/abs/1502.03167) layers were added.** It is known as practical solution for _Bad-Initialization_ and _Vanishing-Gradient_ problems by reducing internal covariance shift between each mini-batch of whole training sequence. As a result, trying to find optimal learning rate was meaningless. Moreover, validation loss was drastically decreased compared to the model without Batch Normalizaiton,.
3) **Changed the activations from RELU to [ELU](https://arxiv.org/abs/1511.07289).** It showed better accuracy in the simulation.
4) Nvidia's model has a lot of parameters so, it is easy to be overfitted. I could see the car driven by model of step 3 tended to be very close to outside - as the result of being overfitted to the data about moving straight. Therefore, **Dropout layer was added right after the Flatten layer.** Drop rate was fixed after a few experiments.
5) The final model (model.py lines: 84-112) is described in the following table.

| Layer | Description |
| ------ | ----- |
| Lambda | Normalize the range of input values between -1 and +1 |
| Cropping2D | To only care about the shape of lane line |
| Conv2D | 5x5x24, strides=(2,2), padding=valid |
| BatchNormalization | |
| Activation | ELU |
| Conv2D | 5x5x36, strides=(2,2), padding=valid |
| BatchNormalization | |
| Activation | ELU |
| Conv2D | 5x5x48, strides=(2,2), padding=valid |
| BatchNormalization | |
| Activation | ELU |
| Conv2D | 3x3x64, strides=(1,1), padding=valid |
| BatchNormalization | |
| Activation | ELU |
| Conv2D | 3x3x64 |
| BatchNormalization | |
| Activation | ELU |
| Flatten | |
| Dropout | Rate = 0.8 |
| Fully Connected | 1164 fan-in, 100 fan-out |
| BatchNormalization | |
| Activation | ELU |
| Fully Connected | 50 fan-out |
| BatchNormalization | |
| Activation | ELU |
| Fully Connected | 10 fan-out |
| BatchNormalization| |
| Fully Connected | 1 fan-out |

#### 2. Opimizer and Training Parameter

1) The model used an **adam optimizer**
2) **Mean square error** is used as loss function.
3) The learning rate was setted to **0.01**.
4) **EPOCH is limited by 5**. The model tends do being overfitted when EPOCH is higher than 5.

#### 3. Training data

1) The training data was created by the training mode of the simulator.
2) It is about the following cases:
 | Case | Purpose|
 | --- | --- |
 | 3-lap of center lane driving | To capture good driving behavior |
 | Driving slowly around curves | To generate large number of data for this case |
 | Recovery from outside to center | To teach how to get back to center |

#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using center lane driving. Here is an example image of center lane driving:

![alt text][image2]

Then I repeated this process on track two in order to get more data points.

To augment the data sat, I also flipped images and angles thinking that this would ... For example, here is an image that has then been flipped:

![alt text][image6]
![alt text][image7]

Etc ....

After the collection process, I had X number of data points. I then preprocessed this data by ...


I finally randomly shuffled the data set and put Y% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was Z as evidenced by ... I used an adam optimizer so that manually training the learning rate wasn't necessary.
