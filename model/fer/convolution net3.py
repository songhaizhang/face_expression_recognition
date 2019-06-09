import os

from keras import Input, Model
from keras.layers import Dense, Dropout, Flatten, Conv2D, BatchNormalization, Activation, MaxPooling2D, SeparableConv2D
from keras.models import Sequential
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint,TensorBoard
from keras_preprocessing.image import ImageDataGenerator
from keras import layers
from keras.layers import GlobalAveragePooling2D

#输入神经网络每一批的数量
batch_size = 128

#图片大小
pic_size = 48

#图片基本路径
base_path = "F:\\bishe\\images"

datagen_train = ImageDataGenerator(
                                   rotation_range=30,
                                   zoom_range=0.5,
                                   horizontal_flip=True)
datagen_validation = ImageDataGenerator(
                                   rotation_range=30,
                                   zoom_range=0.5,
                                   horizontal_flip=True)

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


nb_classes = 7

def big_XCEPTION(input_shape, num_classes):
    img_input = Input(input_shape)
    x = Conv2D(32, (3, 3), strides=(2, 2), use_bias=False)(img_input)
    x = BatchNormalization(name='block1_conv1_bn')(x)
    x = Activation('relu', name='block1_conv1_act')(x)
    x = Conv2D(64, (3, 3), use_bias=False)(x)
    x = BatchNormalization(name='block1_conv2_bn')(x)
    x = Activation('relu', name='block1_conv2_act')(x)

    residual = Conv2D(128, (1, 1), strides=(2, 2),
                      padding='same', use_bias=False)(x)
    residual = BatchNormalization()(residual)

    x = SeparableConv2D(128, (3, 3), padding='same', use_bias=False)(x)
    x = BatchNormalization(name='block2_sepconv1_bn')(x)
    x = Activation('relu', name='block2_sepconv2_act')(x)
    x = SeparableConv2D(128, (3, 3), padding='same', use_bias=False)(x)
    x = BatchNormalization(name='block2_sepconv2_bn')(x)

    x = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)
    x = layers.add([x, residual])

    residual = Conv2D(256, (1, 1), strides=(2, 2),
                      padding='same', use_bias=False)(x)
    residual = BatchNormalization()(residual)

    x = Activation('relu', name='block3_sepconv1_act')(x)
    x = SeparableConv2D(256, (3, 3), padding='same', use_bias=False)(x)
    x = BatchNormalization(name='block3_sepconv1_bn')(x)
    x = Activation('relu', name='block3_sepconv2_act')(x)
    x = SeparableConv2D(256, (3, 3), padding='same', use_bias=False)(x)
    x = BatchNormalization(name='block3_sepconv2_bn')(x)

    x = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)
    x = layers.add([x, residual])
    x = Conv2D(num_classes, (3, 3),
               # kernel_regularizer=regularization,
               padding='same')(x)
    x = GlobalAveragePooling2D()(x)
    output = Activation('softmax', name='out_out')(x)

    model = Model(img_input, output)
    return model


if __name__ == "__main__":
    input_shape = (64, 64, 1)
    num_classes = 7
    model = big_XCEPTION((48, 48, 1), num_classes)
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
