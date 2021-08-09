# Demographer and emotion prediction and face detection
> Predict age, gender and emotion from faces detected in an image.

<img src="images/face_header.jpg">

## Introduction
The aim of this project is to predict information from customers for marketing strategies. How do you know if you are targeting the right customers

## Table of Contents
* [Introduction](#introduction)
* [Technologies Used](#technologies-used)
* [Data Collection](#data-collection)
* [Data Preprocessing](#data-preprocessing)
* [Analysis](#analysis)
* [Conclusion](#conclusion)
* [Challenges](#challenges)
* [Next Steps](#next-steps)


## Technologies Used
- Python 3.8.5
- Numpy 1.19.2
- Tensorflow 2.5.0
- Keras 2.4.3
- Scikit-learn 0.24.2
- Pathlib 2.3.5
- Opencv-python 4.5.2.52
- MTCNN 0.1.0
- Matplotlib 3.3.2
- Seaborn 0.11.0

## Data Collection
1. Age and gender prediction
We used [UTKFace](https://www.kaggle.com/jangedoo/utkface-new) dataset from Kaggle. It contained 20k+ cropped face images from age 1-116 and 5 ethnicities with both male and female. We chose not to include ethnicity in this project because of the limited time frame. 

The below graph showed the disturbution of gender data. 48% of the data is Female and 52% is Male. 
<img src="images/face_header.jpg">

This plot showed the disturbution of age data. This is a skewed dataset with most of the age are between 1-2 and 20-30 years old. So much less for age above 80.
<img src="images/face_header.jpg">

2. Emotion prediction
We used [Emotion Detection](https://www.kaggle.com/ananthu017/emotion-detection-fer) dataset also from Kaggle which contained 35k+ cropped face images with 7 emotions (happiness, neutral, sadness, anger, surprise, disgust, fear).

3. Face detection
Considered of the limited time frame, we didn't train our own model on face detection. Instead, we adapted a pre-trained face detection model, MTCNN, to do the face detection part in this project 

## Data Preprocessing
1. Age
We splited the age into groups ('0-2','3-11', '12-17', '18-24', '25-34', '35-44', '45-54', '55-64', '65-80', '81-116')

## Analysis


## Conclusion


## Challenges


## Next Steps
