import tensorflow as tf
import fer_forward
import os
import fer_generateds
import numpy as np
import time
import fer_config as config

BATCH_SIZE = 128
LEARNING_RATE_BASE = 0.0005
LEARNING_RATE_DECAY = 0.99
REGULARIZER = 1e-4
STEPS = 1000
MOVING_AVERAGE_DECAY = 0.99
train_num_examples=28709
#定义反向传播
def backward():
    # 输入x占位
    x = tf.placeholder(tf.float32, [BATCH_SIZE, config.img_width,
                                    config.img_height, fer_forward.NUM_CHANNELS])
    # 标记y_占位
    y_ = tf.placeholder(tf.float32, [None, fer_forward.OUTPUT_NODE])
    # 获得输出y的前向传播计算图
    y = fer_forward.forward(x,True, REGULARIZER)
    # 定义global_step并初始化为0，不可训练
    global_step = tf.Variable(0, trainable=False)
    # 计算稀疏softmax的交叉熵
    ce = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y, labels=tf.argmax(y_, 1))
    # 交叉熵取平均
    cem = tf.reduce_mean(ce)
    # 损失函数loss含正则化
    loss = cem + tf.add_n(tf.get_collection('losses'))
    # 定义指数衰减学习率
    learning_rate = tf.train.exponential_decay(
        LEARNING_RATE_BASE,
        global_step,
        train_num_examples / BATCH_SIZE,
        LEARNING_RATE_DECAY,
        staircase=True)

    # 采用Adam优化
    train_step = tf.train.AdamOptimizer(learning_rate).minimize(loss, global_step=global_step)
    # 定义滑动平均
    ema = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY, global_step)
    # 将滑动平均作用到所有参数变量
    ema_op = ema.apply(tf.trainable_variables())
    # 每运行一步，所有待优化的参数求滑动平均
    with tf.control_dependencies([train_step, ema_op]):
        train_op = tf.no_op(name='train')

    # 创建一个保存模型的对象
    saver = tf.train.Saver()
    # 判断预测值和标记是否相同
    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    # 定义准确率
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    # 批量获取数据
    img_batch,label_batch=fer_generateds.get_tfrecord(BATCH_SIZE,config.tfRecord_train)
    # 创建一个会话
    with tf.Session() as sess:
        # 变量初始化
        init_op = tf.global_variables_initializer()
        sess.run(init_op)
        # 通过checkpoint文件找到模型文件名
        ckpt=tf.train.get_checkpoint_state(config.MODEL_SAVE_PATH)
        # 如果模型存在
        if ckpt and ckpt.model_checkpoint_path:
            # 加载模型继续训练
            saver.restore(sess,ckpt.model_checkpoint_path)

        # 创建一个线程协调器
        coord=tf.train.Coordinator()
        # 启动入队线程
        threads=tf.train.start_queue_runners(sess=sess,coord=coord)
        for i in range(STEPS):
            xs, ys = sess.run([img_batch,label_batch])
            # reshape 输入数据xs
            reshape_xs = np.reshape(xs, (BATCH_SIZE,
                                         config.img_width,
                                         config.img_height,
                                         fer_forward.NUM_CHANNELS))
            # 训练更新loss,accuracy,step
            _, loss_value, accuracy_value, step = sess.run([train_op, loss, accuracy, global_step],
                                                           feed_dict={x: reshape_xs, y_: ys})
            if (i + 1) % 200 == 0:
                # 输出训练轮数和loss值、accuracy值
                print("%s : After %d training step(s), loss,accuracy on training batch is %g , %g." % (
                time.strftime('%Y-%m-%d %H:%M:%S'), step, loss_value,accuracy_value))
                # 保存模型
                saver.save(sess, os.path.join(config.MODEL_SAVE_PATH, config.MODEL_NAME), global_step=global_step)
        # 终止所有线程
        coord.request_stop()
        coord.join(threads)

def main():
    backward()

if __name__ == '__main__':
    main()


