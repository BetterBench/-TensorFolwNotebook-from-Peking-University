#coding:utf-8
#python 3.5

import tensorflow as tf
import numpy as np
from PIL import Image
import fer_forward
import fer_backward
import fer_config as config

#面部表情类别
classes={0:"Angry",1:"Disgust",2:"Fear",3:"Happy",4:"Sad",5:"Surprise",6:"Neutral"}

#使用保存的模型预测图片数值
def restore_model(testPicArr):
    # 实例化一个数据流图并作为整个 tensorflow 运行环境的默认图
    with tf.Graph().as_default() as tg:
        # 输入x占位
        x = tf.placeholder(tf.float32, [1, config.img_width,
                                        config.img_height, fer_forward.NUM_CHANNELS])
        # 获得输出y的前向传播计算图
        y=fer_forward.forward(x,False,None)
        # 定义预测值为y中最大值的索引号
        preValue=tf.argmax(y,1)
        # 定义滑动平均
        variable_averages=tf.train.ExponentialMovingAverage(fer_backward.MOVING_AVERAGE_DECAY)
        # 将影子变量直接映射到变量的本身
        variables_to_restore=variable_averages.variables_to_restore()
        # 创建一个保存模型的对象
        saver=tf.train.Saver(variables_to_restore)

        # 创建一个会话
        with tf.Session() as sess:
            # 通过checkpoint文件找到模型文件名
            ckpt=tf.train.get_checkpoint_state(config.MODEL_SAVE_PATH)
            # 如果模型存在
            if ckpt and ckpt.model_checkpoint_path:
                # 加载模型
                saver.restore(sess, ckpt.model_checkpoint_path)
                reshape_x = np.reshape(testPicArr, (1,
                                             config.img_width,
                                             config.img_height,
                                             fer_forward.NUM_CHANNELS))
                # 计算预测值
                preValue=sess.run(preValue,feed_dict={x:reshape_x})
                # 返回预测值
                return preValue
            # 如果模型不存在
            else:
                # 输出模型文件未找到提示
                print("No checkpoint file found")
                # 返回-1
                return -1
#图片预处理
def pre_pic(picName):
    img=Image.open(picName)
    # 将图片大小转为config.img_width*config.img_height像素并做平滑处理
    reIm=img.resize((config.img_width,config.img_height),Image.ANTIALIAS)

    # 将图片转为灰度图并存为array
    im_arr=np.array(reIm.convert('L'))
    # 将config.img_width*config.img_height的array拉直
    nm_arr=im_arr.reshape([1,config.img_width*config.img_height])
    # nm_arr内数字类型转换为float32
    nm_arr=nm_arr.astype(np.float32)
    # nm_arr内数字除以255，转为0到1的浮点数
    img_ready=np.multiply(nm_arr,1.0/255.0)
    return img_ready

def application():
    # 输入图片张数提示
    testNum=input("input the number of test pictures:")
    # 对每一个待输入图片
    for i in range(eval(testNum)):
        # 输入图片路径提示
        testPic=input("the path of test picture:")
        # 图片预处理
        testPicArr=pre_pic(testPic)
        # 使用模型预测图片数值
        preValue=restore_model(testPicArr)
        # 输出预测结果
        print("The prediction class is: %s"%classes[preValue[0]])

def main():
    application()

if __name__=="__main__":
    main()





