import numpy as py
import seaborn as sns
from keras.preprocessing.image import load_img,img_to_array
from keras_preprocessing.image import ImageDataGenerator
import  matplotlib.pyplot as plt
import os


#图片大小
pic_size = 48

#图片基本路径
base_path = "F:\\bishe\\images"

plt.figure(0,figsize=(12,20))
cpt = 0

for expression in os.listdir(base_path+"\\train\\"):
    for i in range(1,6):
        cpt = cpt + 1
        plt.subplot(7,5,cpt)
        img = load_img(base_path+"\\train\\"+expression+"\\"+os.listdir(base_path+"\\train\\"+expression)[i],target_size=(pic_size,pic_size))
        plt.imshow(img,cmap="gray")

plt.tight_layout()
#plt.show()

for expression in os.listdir(base_path+"\\train\\"):
    print(str(len(os.listdir(base_path+"\\train\\"+expression)))+" "+expression+" "+"images")


#输入神经网络每一批的数量
batch_size = 128

datagen_train = ImageDataGenerator()
datagen_validation = ImageDataGenerator()

train_generator = datagen_train.flow_from_directory(
    base_path+"\\train",target_size=(pic_size,pic_size),
    color_mode="grayscale",
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=True)

validation_generator = datagen_validation.flow_from_directory(
    base_path+"\\validation",target_size=(pic_size,pic_size),
    color_mode="grayscale",
    batch_size=batch_size,
    class_mode='categorical',
    shuffle = True)
