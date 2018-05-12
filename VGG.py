import tensorflow as tf
import numpy as np

def variable_summaries(var):
    with tf.name_scope('summaries'):
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean', mean)#平均值
        with tf.name_scope('stddev'):
            stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        tf.summary.scalar('stddev', stddev)#标准差
        tf.summary.scalar('max', tf.reduce_max(var))#最大值
        tf.summary.scalar('min', tf.reduce_min(var))#最小值
        tf.summary.histogram('histogram', var)#直方图


def Convolution(x,ksize,stride,filter_out,name,paddingFlag,weight_initializer=None,bias_initializer=None):
    with tf.variable_scope(name) as scope:
        filters_in = x.get_shape()[-1]
        stddev = 1. / tf.sqrt(tf.cast(filter_out, tf.float32))
        if weight_initializer is None:
            weight_initializer = tf.random_uniform_initializer(minval=-stddev, maxval=stddev, dtype=tf.float32)
        if bias_initializer is None:
            bias_initializer = tf.random_uniform_initializer(minval=-stddev, maxval=stddev, dtype=tf.float32)
        shape = [ksize, ksize, filters_in, filter_out]
        # weights = _get_variable('weights',shape, weight_initializer, tf.contrib.layers.l2_regularizer(wd))
        weights = _get_variable('weights', shape, weight_initializer)
        variable_summaries(weights)
        conv = tf.nn.conv2d(x, weights, [1, stride, stride, 1], padding=paddingFlag)
        biases = _get_variable('biases', [filter_out], bias_initializer)
        variable_summaries(biases)
        return tf.nn.bias_add(conv, biases)

def maxPoolLayer(x,ksize,stride,name,padding):
    return tf.nn.max_pool(x, ksize=[1, ksize, ksize, 1],
                          strides=[1, stride, stride, 1], padding=padding, name=name)

def fcLayer(x,input,output,Isrelu,name):
    with tf.variable_scope(name) as scope:
        w = tf.get_variable("w", shape=[input, output], dtype="float")
        b = tf.get_variable("b", [output], dtype="float")
        out = tf.nn.xw_plus_b(x, w, b, name=scope.name)
        if Isrelu:
            return tf.nn.relu(out)
        else:
            return out

def _get_variable(name,shape,initializer, regularizer=None,dtype='float',trainable=True):
    collections = [tf.GraphKeys.GLOBAL_VARIABLES]
    var = tf.get_variable(name,
                          shape=shape,
                          initializer=initializer,
                          dtype=dtype,
                          regularizer=regularizer,
                          collections=collections,
                          trainable=trainable)

    return var


def dropout(x, dropPro, name=None):
    return tf.nn.dropout(x, dropPro, name)

def LRN(x, dr, alpha, beta, name=None, bias=1.0):
    return tf.nn.local_response_normalization(x, depth_radius=dr, alpha=alpha,
                                              beta=beta, bias=bias, name=name)
def flatten(x):
    shape = x.get_shape().as_list()
    dim = 1
    for i in range(1,len(shape)):
        dim*=shape[i]
    return tf.reshape(x, [-1, dim]),dim

def VGG16(input):
    with tf.name_scope("conv_1"):
        conv1 = Convolution(input, 3, 1, 64, "conv1", paddingFlag ="SAME")
    with tf.name_scope("conv_2"):
        conv2 = Convolution(conv1, 3, 1, 64, "conv2", paddingFlag="SAME")
    with tf.name_scope("Maxpool_1"):
        pool2 = maxPoolLayer(conv2, 2, 2, "pool1", "SAME")

    with tf.name_scope("conv_3"):
        conv3 = Convolution(pool2, 3, 1, 128, "conv3", paddingFlag="SAME")
    with tf.name_scope("conv_4"):
        conv4 = Convolution(conv3, 3, 1, 128, "conv4", paddingFlag="SAME")
    with tf.name_scope("Maxpool_2"):
        pool2 = maxPoolLayer(conv4, 2, 2, "pool2", "SAME")

    with tf.name_scope("conv_5"):
        conv5 = Convolution(pool2, 3, 1, 256, "conv5", paddingFlag="SAME")
    with tf.name_scope("conv_6"):
        conv6 = Convolution(conv5, 3, 1, 256, "conv6", paddingFlag="SAME")
    with tf.name_scope("conv_7"):
        conv7 = Convolution(conv6, 3, 1, 256, "conv7", paddingFlag="SAME")
    with tf.name_scope("Maxpool_3"):
        pool3 = maxPoolLayer(conv7, 2, 2, "pool3", "SAME")

    with tf.name_scope("conv_8"):
        conv8 = Convolution(pool3, 3, 1, 512, "conv8", paddingFlag="SAME")
    with tf.name_scope("conv_9"):
        conv9 = Convolution(conv8, 3, 1, 512, "conv9", paddingFlag="SAME")
    with tf.name_scope("conv_10"):
        conv10 = Convolution(conv9, 3, 1, 512, "conv10", paddingFlag="SAME")
    with tf.name_scope("Maxpool_3"):
        pool4 = maxPoolLayer(conv10, 2, 2, "pool4", "SAME")

    with tf.name_scope("conv_11"):
        conv11 = Convolution(pool4, 3, 1, 512, "conv11", paddingFlag="SAME")
    with tf.name_scope("conv_12"):
        conv12 = Convolution(conv11, 3, 1, 512, "conv12", paddingFlag="SAME")
    with tf.name_scope("conv_13"):
        conv13 = Convolution(conv12, 3, 1, 512, "conv13", paddingFlag="SAME")
    with tf.name_scope("Maxpool_5"):
        pool5 = maxPoolLayer(conv13, 2, 2, "pool5", "SAME")

    with tf.name_scope("flatten"):
        fcIn,dim = flatten(pool5)

    with tf.name_scope('fc_1'):
        fc1 = fcLayer(fcIn, dim, 4096, True, "fc1")
        # dropout1 = dropout(fc1, KEEPPRO)

    with tf.name_scope('fc_2'):
        fc2 = fcLayer(fc1, 4096, 4096, True, "fc2")
        # dropout1 = dropout(fc1, KEEPPRO)

    with tf.name_scope('fc_3'):
        fc3 = fcLayer(fc2, 4096, 10, True, "fc3")
        # dropout1 = dropout(fc1, KEEPPRO)

    with tf.name_scope('softmax'):
        prediction = tf.nn.softmax(fc3)

    return prediction
