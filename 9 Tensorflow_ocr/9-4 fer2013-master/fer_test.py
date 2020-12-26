# coding:utf-8
import time
import tensorflow as tf
import fer_forward
import fer_backward
import fer_generateds
import numpy as np
import fer_config as config

TEST_INTERVAL_SECS = 5
TEST_NUM = 3589


def test():
    # 实例化一个数据流图并作为整个 tensorflow 运行环境的默认图
    with tf.Graph().as_default() as g:
        # 输入x占位
        x = tf.placeholder(tf.float32, [TEST_NUM, config.img_width,
                                    config.img_height, fer_forward.NUM_CHANNELS])
        # 标记y_占位
        y_ = tf.placeholder(tf.float32, [None, fer_forward.OUTPUT_NODE])
        # 获得输出y的前向传播计算图
        y = fer_forward.forward(x, False, None)
        # 定义滑动平均
        ema = tf.train.ExponentialMovingAverage(fer_backward.MOVING_AVERAGE_DECAY)
        # 将影子变量直接映射到变量的本身
        ema_restore = ema.variables_to_restore()
        # 创建一个保存模型的对象
        saver = tf.train.Saver(ema_restore)
        # 判断预测值和标记是否相同
        correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
        # 定义准确率
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        # 批量获取测试数据
        img_batch, label_batch = fer_generateds.get_tfrecord(TEST_NUM, config.tfRecord_test)
        for i in range(3):
            # 创建一个会话
            with tf.Session() as sess:
                # 通过checkpoint文件找到模型文件名
                ckpt = tf.train.get_checkpoint_state(config.MODEL_SAVE_PATH)
                # 如果模型存在
                if ckpt and ckpt.model_checkpoint_path:
                    # 加载模型
                    saver.restore(sess, ckpt.model_checkpoint_path)
                    # 获得模型中的global_step
                    global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]

                    # 创建一个线程协调器
                    coord = tf.train.Coordinator()
                    # 启动入队线程
                    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
                    xs, ys = sess.run([img_batch, label_batch])
                    # reshape测试输入数据xs
                    reshape_xs = np.reshape(xs, (TEST_NUM,
												 config.img_width,
												 config.img_height,
                                                 fer_forward.NUM_CHANNELS))
                    # 计算准确率
                    accuracy_score = sess.run(accuracy, feed_dict={x: reshape_xs, y_: ys})
                    # 输出global_step和准确率
                    print("After %s training step(s), test accuracy = %g" % (global_step, accuracy_score))
                    # 终止所有线程
                    coord.request_stop()
                    coord.join(threads)

                else:
                    print('No checkpoint file found')
                    return
            time.sleep(TEST_INTERVAL_SECS)


def main():
    test()


if __name__ == '__main__':
    main()
