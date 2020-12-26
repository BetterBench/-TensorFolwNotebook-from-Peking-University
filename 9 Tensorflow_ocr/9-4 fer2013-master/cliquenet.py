'''
Reference:
Yibo Yang, Zhisheng Zhong, Tiancheng Shen, Zhouchen Lin. Convolutional Neural Networks with Alternately Updated Clique. In CVPR, June 2018.
'''

# 引入tensorflow模块
import tensorflow as tf

NUM_CHANNELS=1
OUTPUT_NODE=7
# 定义获取指定形状权重张量的函数，输入为张量形状、正态分布的方差以及l2正则化的正则化系数
def get_weight(shape, std, regularizer):
    # 使用截止正态分布对张量进行初始化，该正态分布的方差设为函数给定值，并设置为可训练的变量
    w = tf.Variable(tf.truncated_normal(shape, stddev=std))
    # 若设置正则项不为空值，则进行l2正则化
    if regularizer is not None:
    # 正则项不为空，根据函数给定的正则化系数计算该张量的l2范数，加入losses集合，之后会作为结构风险项与经验风险一同构成损失函数
        tf.add_to_collection('losses', tf.contrib.layers.l2_regularizer(regularizer)(w))
    # 返回该权重张量
    return w

# 定义获取指定形状偏置项的函数，输入为该偏置项的形状
def get_bias(shape):
    # 将偏置项均初始化为0，并设置为可训练的变量
    b = tf.Variable(tf.zeros(shape))
    # 返回该偏置项
    return b

# 定义2维卷积模块函数，输入为上一层得到的形如[B,H,W,C]的输入张量、卷积核张量以及卷积的步长
def conv2d(x, w, stride=1):
    # 根据给定参数返回二维卷积后的张量，其中步长设为给定值，并调整padding使得卷积后feature map与原feature map维度成倍数关系
    return tf.nn.conv2d(x, w, strides=[1, stride, stride, 1], padding='SAME')

# 定义全连接层,输入前一层的结点数与后一层的结点数
def fc(x, in_node, out_node, std, regularizer):
    # 获取该层全连接层的权重矩阵
    w = get_weight([in_node, out_node], std, regularizer)
    # 获取该层全连接层的偏置项
    b = get_bias([out_node])
    # 将权重矩阵与输入向量相乘
    x = tf.matmul(x, w)
    # 将乘后获得的向量与偏置项进行相加
    x = x + b
    # 返回经过全连接层后的向量，即对输入向量进行仿射变换
    return x

# 定义批归一化层,输入为上一层得到的张量与是否处在训练阶段标识
def bn(x, train=True):
    # 调用tensorflow封装的批归一化模块，设定滑动平均率为0.9，并在训练阶段对β值进行学习
    x = tf.contrib.layers.batch_norm(x, decay=0.9, scale=True, updates_collections=None, is_training=train)
    # 返回批归一化后的张量
    return x

# 定义dropout,输入为上一层得到的张量与是否处在训练阶段标识
def dropout(x, train):
    # 调用tensorflow封装的dropout模块，设定张量内变量以0.2的几率变为0，在测试阶段则不受影响
    x = tf.layers.dropout(x, rate=0.2, training=train)
    # 返回dropout后的张量
    return x

# 定义了clique块，可以分为stage1与stage2两个阶段，filters参数表示block内每一个连接所需的卷积核个数，layers参数表示block内除输入结点外的结点数
def clique_block(x0, regularizer, in_channel, filters, size, layers, train):
    # 得到了输入结点到该block内其他所有结点的连接卷积核，w0[i]即输入结点指向第i+1个结点的连接卷积核张量
    w0 = [get_weight([size, size, in_channel, filters], 0.1, regularizer) for i in range(layers)]
    # 得到除输入结点外的结点间相互连接的数量，即全排列数A(n,2)
    num_w = layers * (layers - 1)
    # 得到了该block内除输入结点外的其他所有结点的连接卷积核，从第1个结点到第2、3...个结点卷积核依次为w[0]、w[1]...，之后则依次为第2、3...个结点到其他结点的卷积核，在同一个block中的stage1与stage2阶段，卷积核被重复多次利用
    w = [get_weight([size, size, filters, filters], 0.1, regularizer) for i in range(num_w)]
    # 进入block内的stage1阶段
    x = stage1(x0, w0, w, in_channel, filters, layers, train)
    # 进入block内的stage2阶段
    x = stage2(x, w, in_channel, filters, layers, train)
    # 将stage2的结果与该block的输入在channel维度concat在一起，作为该block提取到的的feature输出
    feature = tf.concat([x0, x], axis=3)
    # 返回stage2的结果与feature，其中x用于下一个block的输入，进一步提取更深层的特征，而feature则作为不同block的特征来进行融合，从而用于结果的预测
    return x, feature

# 定义了stage1阶段
def stage1(x0, w0, w, in_channel, filters, layers=5, train=True):
    # 使用输入结点对每一个之外的结点进行初始化
    for i in range(layers):
        # 如果是第一个结点
        if i == 0:
            # 就取w0中的第一个张量作为连接输入结点与第一个结点的卷积核
            weight = w0[i]
            # 第一个结点在stage1只与输入结点相连接
            data = x0
        # 如果不是第一个结点
        else:
            # 就取w0中对应的结点以及w中对应的结点拼成之后使用的卷积核
            weight = tf.concat([w0[i]] + [w[4 * num + i - 1] for num in range(i)], axis=2)
        # 使用拼出的卷积核与前面的结点生成一个新的结点
        x = conv2d(data, weight)
        # 对数据进行批归一化，加快网络训练，减轻梯度消失
        x = bn(x, train)
        # 使用ReLU激活函数使得模型非线性
        x = tf.nn.relu(x)
        # 使用dropout随机使部分结点归零，从而使得模型不易过拟合
        x = dropout(x, train)
        # 将已经生成的结点concat在一起用于生成下一个结点
        data = tf.concat([data, x], axis=3)
    # 除去输入结点与第一个结点(第一个结点在stage2的第一步即需要被更新，因此不需要传入下一阶段)
    _, x = tf.split(data, [in_channel + filters, filters * (layers - 1)], axis=3)
    # 将得到的后面若干个结点传入stage2
    return x

# 定义了stage2模块
def stage2(x, w, in_channel, filters, layers=5, train=True):
    # 对block内每一个结点进行更新
    for i in range(layers):
        # 取w中对应的结点拼成卷积核
        weight = tf.concat([w[4 * num + i - 1] for num in range(i)] + [w[4 * num + i] for num in range(i + 1, layers)], axis=2)
        # 使用拼出的卷积核与最近更新的结点更新最早的结点
        data = conv2d(x, weight)
        # 对数据进行批归一化，加快网络训练，减轻梯度消失
        data = bn(data, train)
        # 使用ReLU激活函数使得模型非线性
        data = tf.nn.relu(data)
        # 使用dropout随机使部分结点归零，从而使得模型不易过拟合
        data = dropout(data, train)
        # 若不为更新最后一个结点
        if i != layers - 1:
            # 则将更新最早的结点去除
            _, x = tf.split(x, [filters, filters * (layers - 2)], axis=3)
        # 将最新更新的结点同其他结点concat在一起，对于非最后结点相当于替换最早更新的结点
        x = tf.concat([data, x], axis=3)
    # 返回经过stage2后的所有经过更新的结点
    return x

# 定义了在多个block之间的转换模块
def transition(x, regularizer, in_channel, out_channel, size, train, use_attention=True):
    # 得到之后进行的1✖1卷积的卷积核张量，输入channel与输出channel为给定值，在该网络中由于较浅，因此暂不使用compression机制，输入channel与输出channel数相等
    w = get_weight([1, 1, in_channel, out_channel], 0.1, regularizer)
    # 根据上述卷积核进行卷积，步长为1，使得feature map维度不变
    x = conv2d(x, w, 1)
    # 对数据进行批归一化，加快网络训练，减轻梯度消失
    x = bn(x, train)
    # 使用ReLU激活函数使得模型非线性
    x = tf.nn.relu(x)
    # 使用dropout随机使部分结点归零，从而使得模型不易过拟合
    x = dropout(x, train)
    # 选择是否使用attention机制，这里采用了channel-wise的attention机制来在转化过程赋予不同channel不同权重，从而使得下一个block的学习效果更好
    if use_attention:
        # 对数据进行global pooling，从而对于对于每一个channel得到一个值
        attention = tf.nn.avg_pool(x, ksize=[1, size, size, 1], strides=[1, size, size, 1], padding='SAME')
        # 将得到的每一个张量转换为一个长度为channel数的长向量
        attention = tf.reshape(attention, [-1, out_channel])
        # 将获得的权重输入一个全连接层，从而对权重进行学习
        attention = fc(attention, out_channel, out_channel // 2, 0.01, regularizer)
        # 使用ReLU激活函数使得attention的学习模块非线性
        attention = tf.nn.relu(attention)
        # 将获得的权重输入一个全连接层，从而对权重进行更深的学习
        attention = fc(attention, out_channel // 2, out_channel, 0.01, regularizer)
        # 使用sigmoid激活函数使得学习到的权重处于0-1的范围内
        attention = tf.sigmoid(attention)
        # 增大张量维度以方便数据与attention的权重的相乘
        attention = tf.expand_dims(attention, 1)
        # 增大张量维度以方便数据与attention的权重的相乘
        attention = tf.expand_dims(attention, 1)
        # 将学习到的channel-wise attention权重分别乘以各自的channel
        x = tf.multiply(x, attention)
    # 对数据进行最大池化,使得feature map的维度减半，使得下一个block可以学习到更深层的信息
    x = tf.nn.avg_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    # 返回经过transition转换后的数据，用于传入下一个block
    return x

# 定义了cliquenet的前向传播函数
def forward(x, train=True, regularizer=None):
    # 得到之后进行卷积的卷积核张量，输入channel为1，输出channel为64，kernel size为3
    w = get_weight([3, 3, 1, 64], 0.1, regularizer)
    # 先进行一次卷积，步长为2，使得feature map维度减半，使得模型提取到浅层特征
    x = conv2d(x, w, 2) 
    # 对数据进行批归一化，加快网络训练，减轻梯度消失
    x = bn(x, train)
    # 使用ReLU激活函数使得模型非线性
    x = tf.nn.relu(x)
    # 对数据进行最大池化,使得feature map的维度减半
    x = tf.nn.max_pool(x, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME')
    # 将数据送入clique block，block内每一个结点的channel数设为36，一个block内除输入结点外有5个结点
    x, feature1 = clique_block(x, regularizer, 64, 36, 3, 5, train)
    # 对第一个block提取到的特征信息进行global pooling
    feature1 = tf.nn.avg_pool(feature1, ksize=[1, 12, 12, 1], strides=[1, 12, 12, 1], padding='SAME')
    # 将上一层的五个更新后结点作为输入，经过transition调整feature map的维度，并利用attention强化特征信息更优的channel，使得下一个block可以更好地利用前层特征信息
    x = transition(x, regularizer, 180, 180, 12, train)
    # 将数据送入clique block，block内每一个结点的channel数设为36，一个block内除输入结点外有5个结点
    x, feature2 = clique_block(x, regularizer, 180, 36, 3, 5, train)
    # 对第二个block提取到的特征信息进行global pooling
    feature2 = tf.nn.avg_pool(feature2, ksize=[1, 6, 6, 1], strides=[1, 6, 6, 1], padding='SAME')
    # 将上一层的五个更新后结点作为输入，经过transition调整feature map的维度，并利用attention强化特征信息更优的channel，使得下一个block可以更好地利用前层特征信息
    x = transition(x, regularizer, 180, 180, 6, train)
    # 将数据送入clique block，block内每一个结点的channel数设为36，一个block内除输入结点外有5个结点
    _, feature3 = clique_block(x, regularizer, 180, 36, 3, 5, train)
    # 对第三个block提取到的特征信息进行global pooling
    feature3 = tf.nn.avg_pool(feature3, ksize=[1, 3, 3, 1], strides=[1, 3, 3, 1], padding='SAME')
    # 将不同层间获取的不同特征进行融合，浅层的特征包含更多的细节信息，而深层特征则包含更多的全局信息
    out = tf.concat([feature1, feature2, feature3], axis=3)
    # 将张量转换为一个长向量，以用于之后的全连接层
    x = tf.reshape(out, [-1, 964])
    # 将向量输入全连接层，从而输出对每一个分类的预测分数
    x = fc(x, 964, 7, 0.01, regularizer)
    # 返回该神经网络的预测结果
    return x
