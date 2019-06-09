import os
from keras.layers import Dense,Dropout,Flatten,Conv2D,BatchNormalization,Activation,MaxPooling2D
from keras.models import Sequential
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint,TensorBoard
from keras_preprocessing.image import ImageDataGenerator

#输入神经网络每一批的数量
batch_size = 128

#图片大小
pic_size = 48

#图片基本路径
base_path = "../input/images/images/"

datagen_train = ImageDataGenerator(
                                   rotation_range=30,
                                   zoom_range=0.5,
                                   horizontal_flip=True)
datagen_validation = ImageDataGenerator(
                                   rotation_range=30,
                                   zoom_range=0.5,
                                   horizontal_flip=True)

train_generator = datagen_train.flow_from_directory(
    base_path+"train",target_size=(pic_size,pic_size),
    color_mode="grayscale",
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=True)

validation_generator = datagen_validation.flow_from_directory(
    base_path+"validation",target_size=(pic_size,pic_size),
    color_mode="grayscale",
    batch_size=batch_size,
    class_mode='categorical',
    shuffle = True)


nb_classes = 7

#初始化cnn
model = Sequential()

#1 - convolution
#64个大小为3*3的卷积滤波器
#输入：单通道 48*48像素图像 -》（48,48,1）张量
model.add(Conv2D(64,(3,3),padding='same',input_shape=(48,48,1),name='op_data'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))

#2nd convolution layer
model.add(Conv2D(128,(5,5),padding='same'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))

#3nd convolution layer
model.add(Conv2D(512,(3,3),padding='same'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))

#4nd convolution layer
model.add(Conv2D(512,(3,3),padding='same'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))

model.add(Flatten())

model.add(Dense(256))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dropout(0.25))

model.add(Dense(512))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dropout(0.25))

model.add(Dense(nb_classes,activation='softmax',name='out_out'))

opt = Adam(lr=0.0001)
model.compile(optimizer=opt,loss='categorical_crossentropy',metrics=['accuracy'])


epochs = 39

checkpoint =ModelCheckpoint("model_weights.h5",monitor='val_acc',verbose=1,save_best_only=True,mode='max')
callbacks_list = [checkpoint]

tbCallBack = TensorBoard(log_dir='./logs',  # log 目录
                 histogram_freq=0,  # 按照何等频率（epoch）来计算直方图，0为不计算
                 batch_size=128,     # 用多大量的数据计算直方图
                 write_graph=True,  # 是否存储网络结构图
                 write_grads=True, # 是否可视化梯度直方图
                 write_images=True,# 是否可视化参数
                 embeddings_freq=0,
                 embeddings_layer_names=None,
                 embeddings_metadata=None)

callbacks_list.append(tbCallBack)

history = model.fit_generator(generator=train_generator,
                              epochs=epochs,
                              validation_data=validation_generator,
                              validation_steps=validation_generator.n//validation_generator.batch_size,
                              steps_per_epoch=train_generator.n//train_generator.batch_size,
                              callbacks=callbacks_list)
