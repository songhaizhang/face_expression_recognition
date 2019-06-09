from keras.models import  load_model,model_from_json
import numpy as np
import cv2

json_file = open('./model.json')
load_model_json = json_file.read()
json_file.close()
model = model_from_json(load_model_json)

model.load_weights('./model_weights.h5')

face_img = cv2.imread('./wo3.jpg',1)


'''
face_img = face_img.reshape(1,48,48,1)
result = model.predict_proba(face_img)
print(result)
'''

fac_img_gray = cv2.cvtColor(face_img,cv2.COLOR_BGR2GRAY)

cascPath ='./haarcascade_frontalface_alt.xml'
cascade = cv2.CascadeClassifier(cascPath)
facelands = cascade.detectMultiScale(fac_img_gray,
                                     scaleFactor=1.1,
                                     minNeighbors=1,
                                     minSize=(30,30))
print(facelands)

if(len(facelands)>0):
    for faceland in facelands:
        x,y,w,h = faceland
        images = []

        image = cv2.resize(fac_img_gray[y:y+h,x:x+w],(48,48))
        print(np.max(image))
       #    image = image/255.0
        cv2.imshow('fac',image)
        cv2.waitKey(0)
        image = image.reshape(1,48,48,1)

        result = np.array([0.0]*7)

        pre_lists = model.predict_proba(image,batch_size=1,verbose=1)
        #result += np.array([pre for pre_list in pre_lists for pre in pre_list])
        print(pre_lists)

