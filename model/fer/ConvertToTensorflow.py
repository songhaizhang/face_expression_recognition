from keras.models import load_model
import os.path as osp
import os
from keras import backend as K, Model
import sys

import tensorflow as tf




output_folder = 'data'

def freeze_session(session, keep_var_names=None, output_names=None, clear_devices=False):
    from tensorflow.python.framework.graph_util import convert_variables_to_constants
    graph = session.graph
    with graph.as_default():
        freeze_var_names = list(set(v.op.name for v in tf.global_variables()).difference(keep_var_names or []))
        output_names = output_names or []
        output_names += [v.op.name for v in tf.global_variables()]
        input_graph_def = graph.as_graph_def()
        if clear_devices:
            for node in input_graph_def.node:
                node.device = ""
        frozen_graph = convert_variables_to_constants(session, input_graph_def,
                                                      output_names, freeze_var_names)
        return frozen_graph

input_fld = sys.path[0]
weight_file = 'model_weights.h5'
output_graph_name = 'tensor_model.pb'

output_fld = input_fld + '/tensorflow_model/'
if not os.path.isdir(output_fld):
    os.mkdir(output_fld)
weight_file_path = osp.join(input_fld, weight_file)

K.set_learning_phase(0)
net_model = load_model(weight_file_path)


print('input is :', net_model.input.name)
print ('output is:', net_model.output.name)

sess = K.get_session()

frozen_graph = freeze_session(K.get_session(), output_names=[net_model.output.op.name])

from tensorflow.python.framework import graph_io

graph_io.write_graph(frozen_graph, output_fld, output_graph_name, as_text=False)

print('saved the constant graph (ready for inference) at: ', osp.join(output_fld, output_graph_name))



'''
#路径参数
input_path = 'F:\Pythonpro\expression recognition'
weight_file = "model_weights.h5"
weight_file_path = osp.join(input_path,weight_file)
output_graph_name = weight_file[:-3]+'.pb'

#转换函数
def keras_to_tensorflow(keras_model,output_dir,model_name,out_prefix="output_",log_tensorboard=True):
    if os.path.exists(output_dir == False):
        os.mkdir(output_dir)

    out_nodes = []

    for i in range(len(keras_model.outputs)):
        out_nodes.append(out_prefix+str(i+1))
        tf.identity(keras_model.output[i],out_prefix+str(i+1))

    sess = K.get_session()

    from tensorflow.python.framework import  graph_io,graph_util

    init_graph = sess.graph.as_graph_def()

    main_graph = graph_util.convert_variables_to_constants(sess,init_graph,out_nodes)

    graph_io.write_graph(main_graph,output_dir,name=model_name,as_text=False)

    if log_tensorboard:
        from tensorflow.python.tools import import_pb_to_tensorboard

        import_pb_to_tensorboard.import_to_tensorboard(
            os.path.join(output_dir,model_name),
            output_dir
        )


'''
'''
def squeezenet_fire_module(input,input_channel_small = 16,input_channel_large = 64):
    channel_axis=3

    input=Conv2D(input_channel_small,(1, 1),padding = "valid")(input)
    input=Activation("relu")(input)

    input_branch_1=Conv2D(input_channel_large,(1, 1),padding = "valid")(input)
    input_branch_1=Activation("relu")(input_branch_1)

    input_branch_2=Conv2D(input_channel_large,(3,3),padding = "same")(input)
    input_branch_2=Activation("relu")(input_branch_2)

    input=concatenate([input_branch_1,input_branch_2],axis=channel_axis)

    return input


def SqueezeNet(input_shape=(48, 48, 1)):

    image_input=Input(shape=input_shape)


    network=Conv2D(64,(3,3),strides = (2,2),padding = "valid")(image_input)
    network=Activation("relu")(network)
    network=MaxPool2D(pool_size = (3,3),strides = (2, 2))(network)

    network=squeezenet_fire_module(input=network,input_channel_small = 16,input_channel_large = 64)
    network=squeezenet_fire_module(input=network,input_channel_small=16,input_channel_large = 64)
    network=MaxPool2D(pool_size=(3,3),strides = (2,2))(network)

    network=squeezenet_fire_module(input=network,input_channel_small = 32,input_channel_large = 128)
    network=squeezenet_fire_module(input=network,input_channel_small = 32,input_channel_large = 128)
    network=MaxPool2D(pool_size=(3,3),strides = (2,2))(network)

    network=squeezenet_fire_module(input=network,input_channel_small = 48,input_channel_large = 192)
    network=squeezenet_fire_module(input=network,input_channel_small = 48,input_channel_large = 192)
    network=squeezenet_fire_module(input=network,input_channel_small = 64,input_channel_large = 256)
    network=squeezenet_fire_module(input=network,input_channel_small = 64,input_channel_large = 256)

    # Remove layers like Dropout and BatchNormalization, they are only needed in training
    # network = Dropout(0.5)(network)

    network=Conv2D(1000,kernel_size = (1,1),padding = "valid",name = "last_conv")(network)
    network=Activation("relu")(network)

    network=GlobalAvgPool2D()(network)
    network=Activation("softmax", name="output")(network)


    input_image=image_input
    model=Model(inputs=input_image,outputs=network)

    return model

keras_model=SqueezeNet()

keras_model.load_weights("squeezenet.h5")


#输出路径
output_dir = osp.join(os.getcwd(),"trans_model")

#加载模型
h5_model = load_model(weight_file_path)

keras_to_tensorflow(h5_model,output_dir=output_dir,model_name=output_graph_name)

print('model saved')
'''
'''
def keras2pb(h5_path='keras.h5', out_folder='data', out_pb ='data.pb'):

    if not os.path.isdir(out_folder):

        os.mkdir(out_folder)

    K.set_learning_phase(0)

    keras_model = load_model(h5_path)


    print('in layer:', keras_model.input.name)

    print('out layer:', keras_model.output.name)

    with K.get_session() as sess:
        frozen_graph = freeze_session(sess, output_names=[keras_model.output.op.name])

        graph_io.write_graph(frozen_graph, out_folder, out_pb, as_text=False)

        print('save keras model as pb file at: ', osp.join(out_folder, out_pb))


'''
