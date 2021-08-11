# Demographer and emotion prediction and face detection
> Predict age, gender and emotion from faces detected in an image.

<img src="images/face_header.jpg">

## Introduction
In this project, we try to provide an alternative way for our clients to collect their customer information. To collect them, we are predicting the age, gender and emotion from a customer face detected in image. Our main target clients are marketing teams in any business(es) since customer data is the key to any marketing strategie(s).   
Our clients could then use the collected data for marketing campaigns or even work as a security system for any age-restricted or gender-restricted venue(s), e.g. casinos.

## Table of Contents
* [Introduction](#introduction)
* [Technologies Used](#technologies-used)
* [Overall Architecture](#overall-architecture)
* [Data Collection](#data-collection)
* [Data Preprocessing](#data-preprocessing)
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

## Overall Architecture
<img src="images/Screenshot%20(111).png">

## Data Collection
1. Age and gender prediction

   We used [UTKFace](https://www.kaggle.com/jangedoo/utkface-new) dataset from Kaggle. It contains 20k+ cropped face images from age 1-116 and 5 ethnicities with both male and female. We chose not to include ethnicity in this project because of the limited time frame. 

   The below graph showed the disturbution of the gender. 48% of the data is Female and 52% is Male. 
   <img src="images/Screenshot%20(104).png">

   This is a skewed dataset in terms of the age with most of them are between 1-2 and 20-30 years old, so much less for age above 80.

   <img src="images/Screenshot%20(109).png">

2. Emotion prediction

   We used [Emotion Detection](https://www.kaggle.com/ananthu017/emotion-detection-fer) dataset which is also from Kaggle and contains 35k+ cropped face images with 7 emotions (happy, neutral, sad, anger, surprise, disgust, fear). 
We only used 3 emotions (happy, neutral, sad) in this project since these 3 emotions is enough to know if a customer is satisfied with the service or product. 

   It is also a skewed dataset with the images of "sad" is much less than the other two emotions.

   dataset | Emotions | no. of images 
   ------- | -------- | ------------- 
   Training | Happy | 7215 |
   -- | Neutral | 4965 |
   -- | Sad | 436 | 
   Testing | Happy | 1774 
   -- | Neutral | 1233 
   -- | Sad | 111 

3. Face detection

   Considered of the limited time frame, we didn't train our own model on face detection. Instead, we used a pre-trained face detection model, MTCNN, to save the time for training another model.

## Data Preprocessing
1. Age and gender

   All the actual age and gender were wriiten on the image file name. For example, "10_0_0_201701102200447314.jpg", the first number "10" means the person in that image was 10 years old. The second number "0" indicated the gender, "0" is male and "1" is female.

   We needed to label each image with the correct information for training our models. We then splited the age into 10 groups (0-2, 3-11, 12-17, 18-24, 25-34, 35-44, 45-54, 55-64, 65-80, 81-116). We did that because it can enhance the prediction accuracy. Also, different age group have different consumption ability or needs and therefore need a different marketing strategies.
   ```
   age = int(file.split('_')[0])
        if age > 80:
            age = 9 #81-116
        elif age > 64:
            age = 8 #65-80
        elif age > 54:
            age = 7 #55-64
        elif age > 44:
            age = 6 #45-54
        elif age > 34:
            age = 5 #35-44
        elif age > 24:
            age = 4 #25-34
        elif age > 17:
            age = 3 #18-24
        elif age > 11:
            age = 2 #12-17
        elif age > 2:
            age = 1 #3-11
        else:
            age = 0 #0-2
   ```

2. Emotion

   ImageDataGenerator (image augmentation) was applied to add more training data into the model and create variability in the data in order to improve the model prediction accuracy. 

## Face detection
**MTCNN**

MTCNN or "Multi-Task Cascaded Convolutional Neural Network" is a python (pip) library written by Github user ipacz (check out the open source project [here](https://github.com/ipazc/mtcnn)). It got its name because of the cascade structure uses in the network and it is "multi-task" because it uses three models. The three models make three types of predictions, face classification, bounding box regression, and facial landmark localization. They are not connected directly; instead, outputs of the previous stage are fed as input to the next stage.


## Model Building (evaluation and validation?)
1. Age 
   Due to the large data size, we randomly selected 5000 images for model training in order to save the training time. 
   Splited the training and testing dataset into 8:2 ratio (training: 4000 images, testing 1000 images).
   result, accuracy: 
   
2. Gender 
  Also randmly selected 5000 images for the gender prediction model with train 8 : test 2 ratio (training: 4000 images, testing 1000 images)

3. Emotion

## Result

## Conclusion


## Challenges
1. Age and emotion dataset are skewed which lead to not accurate prediction

## Next Steps
1. Apply Oversampling and undersampling on our skewed dataset to predict a more accurate result
2. Apply transfer learning on prediction models to enhance accuracy, e.g. VGGFace
3. Crop and save correct labelled data to our training dataset to increase the accuries(how do we measure the accuries?)
4. Apply on real time video
