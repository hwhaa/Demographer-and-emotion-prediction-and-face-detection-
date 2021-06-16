# face detection and prediction model
from matplotlib import pyplot
from matplotlib.patches import Rectangle
from mtcnn.mtcnn import MTCNN
import cv2
from PIL import Image
import numpy as np
from tensorflow.keras.preprocessing import image
from pathlib import Path
from tensorflow.keras.models import model_from_json

# load gender model
f=Path('cnn_gender_rgb.json')
model_structure=f.read_text()
gender_model=model_from_json(model_structure)
gender_model.load_weights('cnn_gender_rgb.h5')
# load age model
f=Path('cnn_age10_rgb.json')
model_structure=f.read_text()
age_model=model_from_json(model_structure)
age_model.load_weights('cnn_age10_rgb.h5')
# load emotion model
f=Path('EMO_MODEL_STRUCTURE.json')
model_structure=f.read_text()
emo_model=model_from_json(model_structure)
emo_model.load_weights('EMO_MODEL_WEIGHTs.h5')

# age and emotion prediction labels
age_labels=['0-2','3-11', '12-17', '18-24', '25-34', '35-44', '45-54', '55-64', '65-80', '81-116']
emo_dict = {0:'Disgust',1:'Happy',2:'Neutral'}

# face detection function
def face_detection(path, faces):
  data=pyplot.imread(path)
  pyplot.imshow(data)
  fig=pyplot.gcf()
  fig.set_size_inches(15,10)
  ax=pyplot.gca()
  for face in faces:
    # coordinates detected by MTCNN
    x,y,width,height=face['box']

    # center alignment 
    center=[x+(width/2),y+(height/2)]
    max_border=max(width,height)
    left=max(int(center[0]-(max_border/2)),0)
    right=max(int(center[0]+(max_border/2)),0)
    top=max(int(center[1]-(max_border/2)),0)
    bottom=max(int(center[1]+(max_border/2)),0)

    # crop the face
    cropped_image_i=data[top:top+max_border,left:left+max_border,:]
    # resize image to fit the age and gender model
    cropped_image=np.array(Image.fromarray(cropped_image_i).resize([64,64]))
    # resize image to fit emo model
    cropped_image_emo=np.array(Image.fromarray(cropped_image_i).resize([48,48]))
    cropped_image_emo=cv2.cvtColor(cropped_image_emo,cv2.COLOR_BGR2GRAY)
    cropped_image_emo=np.expand_dims(cropped_image_emo,0)

    # create predictions
    gender_pred=gender_model.predict(cropped_image.reshape(1,64,64,3))
    age_pred=age_model.predict(cropped_image.reshape(1,64,64,3))
    emo_pred=emo_model.predict(cropped_image_emo.reshape(1,48,48,1))

    # create the box around the face
    rect=Rectangle((left,top),max_border,max_border,fill=False,color='red')
    # add the box
    ax.add_patch(rect)
    
    # add gender prediction
    gender_text='Female' if gender_pred > 0.5 else 'Male'
    ax.text(left,top-(image.shape[0]*0.014),'Gender: {}'.format(gender_text),fontsize=10,color='red')
    # add age prediction
    age_index=int(np.argmax(age_pred))
    age_confident=age_pred[0][age_index]
    age_text=age_labels[age_index]
    ax.text(left,top-(image.shape[0]*0.033),'Age: {}({:.2f})'.format(age_text,age_confident),fontsize=10,color='red')
    # add emotion prediction
    emo_index=np.argmax(emo_pred)
    emo_confident=np.max(emo_pred)
    emo_text=emo_dict[emo_index]
    ax.text(left, top-(image.shape[0]*0.055),'Emotion: {}({:.2f})'.format(emo_text,emo_confident),fontsize=10,color='red')
  # show the resulting image
  pyplot.show()

  
# load the image for detction and prediction
path='./face_detection_images/sadman.jpg'
image=pyplot.imread(path)
detector=MTCNN()
faces=detector.detect_faces(image)
face_detection(path,faces)
