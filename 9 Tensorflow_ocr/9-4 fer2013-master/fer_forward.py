#coding:utf-8
import tensorflow as tf

NUM_CHANNELS=1
CONV1_SIZE=5
CONV1_KERNEL_NUM=64
CONV2_SIZE=5
CONV2_KERNEL_NUM=64
CONV3_SIZE=5
CONV3_KERNEL_NUM=128
FC_SIZE=512
OUTPUT_NODE=7

# 参数
def variable_with_weight_loss(shape,std,w1):
    var = tf.Variable(tf.truncated_normal(shape,stddev=std),dtype=tf.float32)
    if w1 is not None:
        weight_loss = tf.multiply(tf.nn.l2_loss(var),w1,name="weight_loss")
        tf.add_to_collection("losses",weight_loss)
    return var

# 卷积层
def conv2d(x, w):
    return tf.nn.conv2d(x, w, strides=[1, 1, 1, 1], padding='SAME')

#池化层
def max_pooling_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME')

# 前向传播网络
def forward(x, train,regularizer):

    # 第一层卷积
    conv1_w = variable_with_weight_loss(shape=[CONV1_SIZE,CONV1_SIZE,NUM_CHANNELS,CONV1_KERNEL_NUM],std=5e-2,w1=0)
    conv1_b = tf.Variable(tf.constant(0.0,dtype=tf.float32,shape=[CONV1_KERNEL_NUM]))
    conv1 = conv2d(x, conv1_w)
    relu1 = tf.nn.relu(tf.nn.bias_add(conv1, conv1_b))
    pool1 = max_pooling_2x2(relu1)
    # 局部响应归一化
    norm1 = tf.nn.lrn(pool1, 4, bias=1.0, alpha=0.001 / 9, beta=0.75)

    # 第二层卷积
    conv2_w = variable_with_weight_loss(shape=[CONV2_SIZE,CONV2_SIZE,CONV1_KERNEL_NUM,CONV2_KERNEL_NUM],std=5e-2,w1=0)
    conv2_b = tf.Variable(tf.constant(0.1,dtype=tf.float32,shape=[CONV2_KERNEL_NUM]))
    conv2 = conv2d(norm1, conv2_w)
    relu2 = tf.nn.relu(tf.nn.bias_add(conv2, conv2_b))
    norm2 = tf.nn.lrn(relu2, 4, bias=1.0, alpha=0.01 / 9, beta=0.75)
    pool2 = max_pooling_2x2(norm2)

    # 第三层卷积
    conv3_w = variable_with_weight_loss(shape=[CONV3_SIZE,CONV3_SIZE,CONV2_KERNEL_NUM,CONV3_KERNEL_NUM],std=5e-2,w1=0)
    conv3_b = tf.Variable(tf.constant(0.1,dtype=tf.float32,shape=[CONV3_KERNEL_NUM]))
    conv3 = conv2d(pool2, conv3_w)
    relu3 = tf.nn.relu(tf.nn.bias_add(conv3, conv3_b))
    norm3 = tf.nn.lrn(relu3, 4, bias=1.0, alpha=0.01 / 9, beta=0.75)
    pool3 = max_pooling_2x2(norm3)
    if train: pool3 = tf.nn.dropout(pool3, 0.95)

    ##全连接层
    pool_shape = pool3.get_shape().as_list()
    nodes = pool_shape[1] * pool_shape[2] * pool_shape[3]
    reshaped = tf.reshape(pool3, [pool_shape[0], nodes])

    w1 = variable_with_weight_loss([nodes, FC_SIZE], std=0.04, w1=0.004)
    b1 = tf.Variable(tf.constant(0.1,shape=[FC_SIZE],dtype=tf.float32))
    y1 = tf.nn.relu(tf.matmul(reshaped, w1) + b1)
    if train: y1 = tf.nn.dropout(y1, 0.95)

    w2=variable_with_weight_loss([FC_SIZE, FC_SIZE], std=0.04, w1=0.004)
    b2 = tf.Variable(tf.constant(0.1,shape=[FC_SIZE],dtype=tf.float32))
    y2 = tf.nn.relu(tf.matmul(y1, w2) + b2)
    if train: y2 = tf.nn.dropout(y2, 0.95)

    w3 = variable_with_weight_loss([FC_SIZE, OUTPUT_NODE], std=1.0 / FC_SIZE, w1=0)
    b3 = tf.Variable(tf.zeros([OUTPUT_NODE]))
    y = tf.matmul(y2, w3) + b3
    return y

