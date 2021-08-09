# Demographer and emotion prediction and face detection
> Predict age, gender and emotion from faces detected in an image.

<img src="images/face_header.jpg">

## Introduction
In this project, we try to provide an alternative way for our clients to collect their customer information. To collect them, we are predicting the age, gender and emotion from a customer face detected in image. Our main target clients are marketing teams in any business(es) since customer data is the key to any marketing strategie(s).   
Our clients could use the data that is collected for marketing campaigns, smart marketing 

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
We used [Emotion Detection](https://www.kaggle.com/ananthu017/emotion-detection-fer) dataset which is also from Kaggle which contained 35k+ cropped face images with 7 emotions (happy, neutral, sad, anger, surprise, disgust, fear). We were only using 3 emotions (happy, neutral, sad) since it only need to know 

3. Face detection
Considered of the limited time frame, we didn't train our own model on face detection. Instead, we adapted a pre-trained face detection model, MTCNN, to do the face detection part in this project 

## Data Preprocessing
1. Label the images
All the actual age and gender were wriiten on the image file name. For example, "10_0_0_201701102200447314.jpg", the first number "10" means the person in that image was 10 years old. The second number "0" indicated the gender, "0" is male and "1" is female.

After we labelled the images, we splited the age into 10 groups (0-2, 3-11, 12-17, 18-24, 25-34, 35-44, 45-54, 55-64, 65-80, 81-116).
```

```

2. 

## Overall Architecture

## Model Building

## Conclusion


## Challenges
1. Age and emotion dataset are skewed which lead to not accurate prediction

## Next Steps
1. Apply Oversampling and undersampling on our skewed dataset to predict a more accurate result
2. Apply transfer learning on prediction models to enhance accuracy, e.g. VGGFace
3. Crop and save correct labelled data to our training dataset to increase the accuries(how do we measure the accuries?)
4. Apply on real time video
