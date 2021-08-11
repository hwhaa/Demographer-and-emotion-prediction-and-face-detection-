# Demographer and emotion prediction and face detection
> Predict age, gender and emotion from faces detected in an image.

<img src="images/face_header.jpg">

## Introduction
In this project, we are trying to provide an alternative way for our clients to collect their customers' information. Through the image inputted by the user, we detect the faces and predict the age, gender and emotion. 

Our main target clients are marketing teams in any business since customer data is the key to any marketing strategy. Apart from that, even clients from the secrity industry can use it as a security system for any age-restricted or gender-restricted venue, e.g. casinos.

## Table of Contents
* [Introduction](#introduction)
* [Technologies Used](#technologies-used)
* [Overall Architecture](#overall-architecture)
* [Data Collection](#data-collection)
* [Data Preprocessing](#data-preprocessing)
* [Face Detection Model](#face-detection-model)
* [Prediction Models Building](#prediction-models-building)
* [Result](#result)
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

   We used [UTKFace](https://www.kaggle.com/jangedoo/utkface-new) dataset from Kaggle. It contains 20k+ cropped face images of both male and female from age 1-116 and 5 ethnicities. We chose not to include ethnicity in this project because of the limited time frame. 

   The below graph showed the disturbution of the gender. 48% of the data is Female and 52% is Male. 
   <img src="images/Screenshot%20(104).png">

   This is a skewed dataset in terms of the age with most of them are between 1-2 and 20-30 years old, so much less for age above 80.

   <img src="images/Screenshot%20(109).png">

2. Emotion prediction

   We used [Emotion Detection](https://www.kaggle.com/ananthu017/emotion-detection-fer) dataset which is also from Kaggle and contains 35k+ cropped face images with 7 emotions (happy, neutral, sad, anger, surprise, disgust, fear). 
We only used 3 emotions (happy, neutral, disgust) in this project since these 3 emotions is enough to know if a customer is satisfied with the service or product. 

   It is a skewed dataset with "disgust" has much less images than the other two emotions.

   dataset | Emotions | no. of images 
   ------- | -------- | ------------- 
   Training | Happy | 7215 |
   -- | Neutral | 4965 |
   -- | Disgust | 436 | 
   Testing | Happy | 1774 
   -- | Neutral | 1233 
   -- | Disgust | 111 

3. Face detection

   Considered of the limited time frame, we didn't train our own model on face detection. Instead, we used a pre-trained face detection model, MTCNN, to save the time for training another model.

## Data Preprocessing
1. Labelling

   All the actual age and gender were written on the image file name. For example, "10_0_0_201701102200447314.jpg", the first number "10" means the person in that image was 10 years old. The second number "0" indicated the gender, "0" is male and "1" is female.

   We needed to label each image with the correct information for models building. After that, we splited the age into 10 groups (0-2, 3-11, 12-17, 18-24, 25-34, 35-44, 45-54, 55-64, 65-80, 81-116). We did that because it can enhance the prediction accuracy. Also, different age group have different consumption ability or needs and therefore need a different marketing strategy.
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

2. Image Augmentation

   ImageDataGenerator (image augmentation) was applied to create variability in the emotion dataset in order to improve the model prediction accuracy. 

## Face Detection Model
**MTCNN**

MTCNN or "Multi-Task Cascaded Convolutional Neural Network" is a python (pip) library written by Github user ipacz (check out the open source project [here](https://github.com/ipazc/mtcnn)). It got its name because of the cascade structure uses in the network and it is "multi-task" because it uses three models. The models make three types of predictions, face classification, bounding box regression and facial landmark localization. They are not connected directly; instead, outputs of the previous stage are fed as input to the next stage.


## Prediction Models Building
To be able to make accurate prediction, we built CNN models to extract featrues from the images.

1. Age 

   Due to the large data size, we randomly selected 5000 images for model building in order to save training time. 
   After that we categorised them into 10 age group and splited the training and testing dataset into 8:2 ratio (training: 4000 images, testing 1000 images).
   
   <img src="images/Screenshot%20(61).png">
   
   The age model gave around 85% validation accuracy after 20 epoches.
   
2. Gender 

   Started with randomly selected 5000 images for the gender prediction model with 8:2 train test ratio (training: 4000 images, testing 1000 images).
   
   <img src="images/Screenshot%20(65).png" width="450" height="300">
   
   Both the training and validation accuracies were low with only around 47-50%. 

3. Emotion

   Training set had 12616 images and testing set had 3118 images, giving a train test ratio of 8:2. Image augmentation was then applied on both training and testing dataset.

   <img src="images/Screenshot%20(114).png" width="450" height="300">
   
   The validation accuracy were around 84% and slightly higher than the training accuracy which might due to heavy dropout and imbalance dataset(most are "happy" and "neutral").
   
## Result
After we have face detection model, age, gender and emotion prediction models ready, it is time to apply.

Here is the input image with 8 people. 

<img src="images/group_picture.jpg" width="550" height="370">

The MTCNN model can accurately locate the 8 faces and put bounding boxes around them.

<img src="images/Screenshot%20(67).png" width="600" height="370">

We then apply all three prediction models on the detected faces. 

<img src="images/Screenshot%20(68).png" width="170" height="180">

However, we discovered that the age model performed pretty bad especially on the elderly. Many of them were predicted as age 0-2 or 3-11. 

<img src="images/Screenshot%20(106).png">

Also, it was not ideal when predicting the "disgust" emotion, most of them were predicted as "neutral".

<img src="images/Screenshot%20(107).png">


## Challenges
1. Skewed datasets

   Age prediction model had a very low validation accuracies of 47%. It was mainly beacuse age 20-30 had made up a big part of the dataset when age above 80 had such a tiny part. The impact were showed clearly when we applied the model on images, it couldn't make accurate prediction on elderly. 
   
   Same as emotion model, the lack of "disgust" emotion data led to a bad performance on "disgust face".
   
2. Long training time
   
   CNN could be very computational expensive with all the hidden layers involved to extract features. Long training time plus limited project time frame, we downsized the age and gender dataset from 20k+ to only 5000 images. It could be one of the reasons why age prediction model had low accuracy because there was not enough data to train for predicting 10 classes.


## Next Steps
1. Apply oversampling and undersampling on our dataset to balance out the skewed situation
2. Try different hyperparameters tuning when training models
3. Apply transfer learning on prediction models to enhance accuracy, e.g. VGGFace
4. Apply on real time video
5. Deploy on streamlit or other app
