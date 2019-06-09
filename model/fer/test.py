import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf

import  numpy as np

import matplotlib.pyplot as plt



#读取图片并展示
import cv2


#1.文件的读取 2.封装格式的解析 3.数据解析 4.数据加载

#参数：1名称 2还有0代表gray 1代表color
#img = cv2.imread('C:/Users/nanfengchuiyeluo/Desktop/sucai1.jpg',1)



#参数：1窗体的名称 2展示的内容

#cv2.imwrite('C:/Users/nanfengchuiyeluo/Desktop/sucai2.jpg',img,[cv2.IMWRITE_JPEG_QUALITY,50])


#cv2.imshow('image',img)

#cv2.waitKey(0)


#像素的读写
'''

(b,g,r) = img[100,100]
print(b,g,r)

for i in range(1,100):
    img[10+i,100+i] = (0,0,255)

cv2.imshow('image',img)
cv2.waitKey(0)


data1 = tf.constant(2,dtype=tf.int32)
data2 = tf.Variable(100,name='var')
print(data1)
print(data2)

sess = tf.Session()
init = tf.global_variables_initializer()

print(sess.run(data1))
sess.run(init)
print(sess.run(data2))
sess.close()


#若觉得麻烦则使用with
with sess:
    sess.run(init)
    print(sess.run(data2))


data1 = tf.constant(10)
data2 = tf.Variable(3)

dataAdd = tf.add(data1,data2)
dataCopy = tf.assign(data2,dataAdd)
dataCopy2 = dataAdd
dataSub = tf.subtract(data1,data2)
dataMul = tf.multiply(data1,data2)
dataDiv = tf.divide(data1,data2)
init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    print(sess.run(dataAdd))
    print(sess.run(dataSub))
    print(sess.run(dataMul))
    print(sess.run(dataDiv))
    print('datacopy',sess.run(dataCopy))
    print('datacopy2',sess.run(dataCopy2))
    print(dataCopy.eval())
    print(dataCopy.eval())
    print(dataCopy.eval())

print('end!')


data1 = tf.placeholder(tf.float32)
data2 = tf.placeholder(tf.float32)
dataAdd = tf.add(data1,data2)
with tf.Session() as sess:
    print(sess.run(dataAdd,feed_dict={data1:3,data2:4}))
print('end!')


data1 = tf.constant([[1]])

data2 = tf.constant([[1,2],
                     [4,5]])

dataMut = tf.multiply(data2,data1)

with tf.Session() as sess:
    print(sess.run(dataMut)) #打印一行
    #print(sess.run(data[:,1])) #打印一列
    #print(sess.run(data[1,0]))#打印某行某列

#初始化为0
mat1 = tf.zeros([2,3])
#初始化为1
mat2 = tf.ones([2,3])
#初始化为特定值
mat3 = tf.fill([2,3],10)
#初始化为像mat4的零矩阵
mat4 = tf.constant([[1],[2],[3]])
mat5 = tf.zeros_like(mat4)
#分11等份
mat6 = tf.linspace(0.0,2.0,11)
#随机矩阵
mat7 = tf.random_uniform([2,3],1,10)
with tf.Session() as sess:
    print(sess.run(mat1))
    print(sess.run(mat2))
    print(sess.run(mat3))
    print(sess.run(mat5))
    print(sess.run(mat6))
    print(sess.run(mat7))


#画折现
x = np.array([1,3,5,7,8,9])
y = np.array([15,20,35,24,46,27])
plt.plot(x,y,'r')
#plt.show()
#画柱状
x = np.array([1,3,5,7,8,9])
y = np.array([15,20,35,24,46,27])
plt.bar(x,y,0.5,alpha=1,color='b') #0.5是底边占的比例，alpha是透明度
plt.show()


plt.subplot(2,2,1)
plt.suptitle('optimator',fontsize=10)
plt.ylabel('loss',fontsize=10)
plt.legend(loc='upper right')

plt.subplot(2,2,2)
plt.ylabel('acc',fontsize=10)
plt.legend(loc='upper left')
plt.show()


class model:

    if __name__ == '__main__':
        print("hhh")

        data = np.linspace(1, 15, 15)
        endprice = np.array(
            [2511.90, 2538.26, 2510.68, 2591.66, 2732.98, 2701.69, 2701.29, 2678.67, 2726.50, 2601.50, 2739.17, 2715.07,
             2823.58, 2864.90, 2919.00])
        beginprice = np.array(
            [2438.71, 2500.88, 2534.95, 2512.52, 2594.04, 2743.26, 2697.47, 2695.24, 2678.23, 2722.13, 2674.93, 2744.13,
             2717.46, 2832.73, 2877.40])
        print(data)
        plt.figure()

        for i in range(0, 15):
            dataone = np.zeros([2])
            dataone[0] = i
            dataone[1] = i
            priceone = np.zeros([2])
            priceone[0] = beginprice[i]
            priceone[1] = endprice[i]
        if endprice[i] > beginprice[i]:
            plt.plot(dataone, priceone, 'r', lw=8)  # priceone是两个值
        else:
            plt.plot(dataone, priceone, 'g', lw=8)
        # plt.show()

        dataNormal = np.zeros([15, 1])
        priceNormal = np.zeros([15, 1])
        for i in range(0, 15):
            dataNormal[i, 0] = i / 14.0
            priceNormal[i, 0] = endprice[i] / 3000.0
        x = tf.placeholder(tf.float32, [None, 1])
        y = tf.placeholder(tf.float32, [None, 1])

        w1 = tf.Variable(tf.random_uniform([1, 10], 0, 1))
        b1 = tf.Variable(tf.zeros([1, 10]))
        wb1 = tf.matmul(x, w1) + b1
        layer1 = tf.nn.relu(wb1)

        w2 = tf.Variable(tf.random_uniform([10, 1], 0, 1))
        b2 = tf.Variable(tf.zeros([15, 1]))
        wb2 = tf.matmul(layer1, w2) + b2
        layer2 = tf.nn.relu(wb2)

        loss = tf.reduce_mean(tf.square(y - layer2))
        train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss)

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            for i in range(0, 10000):
                sess.run(train_step, feed_dict={x: dataNormal, y: priceNormal})  # x也为一个n维1列的矩阵，相当于直接把datanormal直接喂给x
            pred = sess.run(layer2, feed_dict={x: dataNormal})  # 此时参数都已调整好，再喂入x
            predPrice = np.zeros([15, 1])
            # predPrice2 = np.zeros([15, 1])
            for i in range(0, 15):
                predPrice[i, 0] = (pred * 3000)[i, 0]  # 赋值操作，首先先*3000再赋值
            # predPrice2[i,0] = pred[i,0]*3000
            plt.plot(data, predPrice, 'b', lw=1)
        plt.show()
'''
'''

#图片的缩放
img = cv2.imread('./sucai1.jpg',1)
imginfo = img.shape
print(imginfo)
height = imginfo[0]
width = imginfo[1]
mode = imginfo[2]

dstheight = int(height*0.5)
dstwidth = int(width*0.5)
dst = cv2.resize(img,(dstwidth,dstheight))
cv2.imshow('fac',dst)
cv2.waitKey(0)
'''

#最近临域插值  双线性差值
#src 10*20 dst 5*10
#dst <-src
#(1,2) <- (2,4)
#dst x 1 -> src x 2 newx
#newx = x*(src行/目标行) newx = 1*(10/5) = 2  (x为目标的x，newx代表原图像的x)
#newy = y*(src列/目标列) newy =2*(20/10) = 4 (y为目标的y)
#若计算出的值为小数，如12.3 = 12，取整

#双线性插值
#A1 = 20% 上+80%下 A2同理
#B1 = 30% 左+70%右 B2同理
#第一种计算方法：最终点 = A1 30%+A2 70%
#第二种计算方法：最终点 = B1 20%+B2 80%



