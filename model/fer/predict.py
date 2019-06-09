import cv2
import os
import  numpy as np
import sys
from keras.models import load_model




def detect_face(image_path):
    cascPath ='./haarcascade_frontalface_alt.xml'
    faceCasccade = cv2.CascadeClassifier(cascPath)

    #load the img and convert it to bgrgray
    img = cv2.imread(image_path)
    img_gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    faces =faceCasccade.detectMultiScale(img_gray,
                                         scaleFactor=1.1,
                                         minNeighbors=1,
                                         minSize=(30,30))

    return  faces,img_gray,img

def predict_emotion(face_img):
    face_img = face_img
    resize_img = cv2.resize(face_img,(48,48))
    rsz_img = []
    rsh_img = []
    results = []
    rsz_img.append(resize_img[:,:])
    rsz_img.append(resize_img[2:45,:])
    rsz_img.append(cv2.flip(rsz_img[0],1))

    i = 0
    for rsz_image in rsz_img:
        rsz_img[i] = cv2.resize(rsz_image,(48,48))

        i += 1;
    for rsz_image in rsz_img:
        rsh_img.append(rsz_image.reshape(1,48,48,1))
    i=0
    for rsh_image in rsh_img:
        list_of_list = model.predict(rsh_image,batch_size=32,verbose=1)
        result = [prob for lst in list_of_list for prob in lst]
        results.append(result)
    return results

if __name__=='__main__':
    model =load_model('./model_weights.h5')
    face_img = './sucai1.jpg'

    faces, img_gray, img = detect_face(face_img)

    for (x,y,w,h) in faces:
        face_img_gray = img_gray[y:y + h, x:x + w]

        results = predict_emotion(face_img_gray)
        result_sum = np.array([0] * 7)
        for result in results:
            result_sum = result_sum + np.array(result)
            print("这里是result--------------------")
            print(result)
         










