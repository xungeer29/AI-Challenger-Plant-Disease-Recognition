# -*- coding:utf-8 -*-

import json
import tensorflow as tf
import os
import numpy as np
import random
from tensorflow.python.platform import gfile

# 32739
TRAIN_IMAGES_DIR = '/data2/gaofuxun/data/Plant_Disease_Recognition/\
ai_challenger_pdr2018_trainingset_20180905/AgriculturalDisease_trai\
ningset/images/'

# 4982
VAL_IMAGES_DIR = '/data2/gaofuxun/data/Plant_Disease_Recognition/\
ai_challenger_pdr2018_validationset_20180905/AgriculturalDisease_\
validationset/images/'

# 4959
TEST_IMAGES_DIR = '/data2/gaofuxun/data/Plant_Disease_Recognition/\
AgriculturalDisease_testA/images/'

JSON_TRAIN = '/data2/gaofuxun/data/Plant_Disease_Recognition/\
ai_challenger_pdr2018_trainingset_20180905/AgriculturalDisease_trai\
ningset/AgriculturalDisease_train_annotations.json'

JSON_VAL = '/data2/gaofuxun/data/Plant_Disease_Recognition/\
ai_challenger_pdr2018_validationset_20180905/AgriculturalDisease_\
validationset/AgriculturalDisease_validation_annotations.json'

log_dir = './log/'
model_dir = './model/'
bottleneck_path = '/data2/gaofuxun/data/Plant_Disease_Recognition/\
bottleneck/'
bottleneck_train_dir = 'train/'
bottleneck_val_dir = 'val/'
bottleneck_test_dir = 'test/'

if not os.path.exists(log_dir):
    os.makedirs(log_dir)
if not os.path.exists(model_dir):
    os.makedirs(model_dir)
if not os.path.exists(bottleneck_path+bottleneck_train_dir):
    os.makedirs(bottleneck_path+bottleneck_train_dir)
if not os.path.exists(bottleneck_path+bottleneck_val_dir):
    os.makedirs(bottleneck_path+bottleneck_val_dir)
if not os.path.exists(bottleneck_path+bottleneck_test_dir):
    os.makedirs(bottleneck_path+bottleneck_test_dir)


BATCH_SIZE = 64 # 设为2的指数倍
# IMAGE_SIZE = 128 # 设为2的指数倍 使用inception-v3不需要指定图像大小
LEARNING_RATE = 0.01
# 衰减系数
DECAY_RATE = 0.9
# 衰减间隔数
DECAY_STEPS = 100
STEPS = 140000
CLASSES = 61

BOTTLENECK_TENSOR_SIZE = 2048
# 瓶颈层张量名
BOTTLENECK_TENSOR_NAME = 'pool_3/_reshape:0' # [1, 1, 1, 2048]
# 图像输入层张量名
JPEG_DATA_TENSOR_NAME = 'DecodeJpeg/contents:0'

# inception_v3 路径
INCEPTION_DIR = './inception_v3/tensorflow_inception_graph.pb'


"""
使用Inecption-v3处理一张图像，返回该图像的feature map
"""
def run_bottleneck_on_image(sess, image, image_data_tensor, bottleneck_tensor):
    bottleneck_values = sess.run(bottleneck_tensor,
                        {image_data_tensor: image})
    print('bottleneck shape:', bottleneck_values.shape)
    # 将四维张量转化成一维数组
    bottleneck_values = np.squeeze(bottleneck_values)

    return bottleneck_values

"""
获取一张图像经过inception-v3处理后的Tensor
如果没有则先计算保存
input:
    --image_name: 图像名
    --image_path: 图像路径
    --category: train val test
    --jpeg_data_tensor: 
return:
    --bottleneck_values: 图像经inception-v3后的bottleneck输出的特征向量
"""
def get_or_create_bottleneck(sess, image_name, image_path, category, 
                             jpeg_data_tensor, bottleneck_tensor):
    if not image_name+'.txt' in os.listdir(bottleneck_path+category+'/'):
        image_dir = image_path + image_name
        image = gfile.FastGFile(image_dir, 'rb').read()
        image = tf.image.decode_jpeg(image)
        print image.eval()
        # 使用数据增强
        image = data_argument(image)
        # 可视化增强的图像
        tf.summary.image('data_argument', image, max_images=9)
        bottleneck_values = run_bottleneck_on_image(sess, image, 
                                jpeg_data_tensor, bottleneck_tensor)

        # 将特征向量写入txt
        bottleneck_string = ','.join(str(x) for x in bottleneck_values)
        with open(bottleneck_path+category+'/'+image_name+'.txt', 'w') as f:
            f.write(bottleneck_string)
    else:
        try:
            # 直接从文件中读取保存的特征向量
            # bottleneck_txt = bottleneck_path+category+'/'+image_name+'.txt'
            # bottleneck_txt = bottleneck_txt.decode('unicode-escape')
            with open(bottleneck_path+category+'/'+image_name+'.txt', 'r') as f:
                bottleneck_string = f.read()
            bottleneck_values = [float(x) for x in bottleneck_string.split(',')]
        except:
            print('Faile to read feature file, re-compute: ', image_name)
            os.remove(bottleneck_path+category+'/'+image_name+'.txt')
            get_or_create_bottleneck(sess, image_name, image_path, category,
                                     jpeg_data_tensor, bottleneck_tensor)

    return bottleneck_values


"""
随机划分出一个batch的数据
Input:
    --json_dir: json文件地址
    --image_path: 图像路径 跟选择train val test模式有关
    --category：train val test
    --jpeg_data_tensor:inception的输入层，使用inception时用到
    --bottleneck_tensor: inception的bottleneck输出层
return:
    --bottlenencks：一batch图像的bottleneck输出特征向量
    --groudtruths：该batch图像的groundtruth
"""
def get_batch_images(sess, json_dir, image_path, category, jpeg_data_tensor,
                     bottleneck_tensor):
    with open(json_dir, 'r') as f:
        image_label_list = json.load(f)

    bottlenecks = []
    groundtruths = []
    for _ in range(BATCH_SIZE):
        # 随机获取一张图像和标签的字典
        index = random.randrange(len(image_label_list))
        image_label_dict = image_label_list[index]

        # 得到图像经inception后的特征向量
        image_name = image_label_dict['image_id']
        image_name = image_name.encode('utf-8')
        bottleneck_values = get_or_create_bottleneck(sess, image_name,
                image_path, category, jpeg_data_tensor, bottleneck_tensor)
        bottlenecks.append(bottleneck_values)

        # 将label进行onehot编码
        label = image_label_dict['disease_class']
        groundtruth = np.zeros(CLASSES, dtype=np.float32)
        groundtruth[int(label)] = 1.0
        groundtruths.append(groundtruth)

    return bottlenecks, groundtruths

"""
数据增强
"""
def data_argument(image):
    try:
        # 随机裁剪
        rate = np.random.randint(8, 10)
        image_size = tf.cast(tf.shape(image).eval(), tf.int32)
        image = tf.random_crop(image, [int(image_size[0]*rate*0.1), int(image_size[1]*rate*0.1), 3])
    except:
        print tf.shape(image).eval()
    # 随机左右翻转
    image = tf.image.random_flip_left_right(image)
    # 随机上下翻转
    image = tf.image.random_up_down(image)
    # 随机旋转90*n次
    image = tf.image.rot90(image, np.random.randint(1, 4))
    # 均值变为0,方差变为1
    image = tf.image.per_image_whitening(image)
    
    return image 


"""
获取全部val数据，计算正确率
"""
def get_val_bottlenecks(sess, json_dir, image_path, jpeg_data_tensor, 
                        bottleneck_tensor):
    with open(json_dir, 'r') as f:
        image_label_list = json.load(f)

    bottlenecks = []
    groundtruths = []
    category = 'val'
    for index in range(len(image_label_list)):
        image_label_dict = image_label_list[index]

        # 得到图像经inception后的特征向量
        image_name = image_label_dict['image_id']
        image_name = image_name.encode('utf-8')
        bottleneck_values = get_or_create_bottleneck(sess, image_name,
                image_path, category, jpeg_data_tensor, bottleneck_tensor)
        bottlenecks.append(bottleneck_values)

        # 将label进行onehot编码
        label = image_label_dict['disease_class']
        groundtruth = np.zeros(CLASSES, dtype=np.float32)
        groundtruth[int(label)] = 1.0
        groundtruths.append(groundtruth)

    return bottlenecks, groundtruths

def get_test_bottlenecks(sess, image_path, jpeg_data_tensor, bottleneck_tensor):
    bottlenecks = []
    images_name = []
    category = 'test'
    for image_name in os.listdir(image_path):
        bottleneck_values = get_or_create_bottleneck(sess, image_name,
                image_path, category, jpeg_data_tensor, bottleneck_tensor)
        bottlenecks.append(bottleneck_values)
        images_name.append(image_name)

    return bottlenecks, images_name

"""
fine-tuning:
自己搭建几层fc增强分类效果
"""
def model(bottleneck_input):
    with tf.name_scope('fc1') as scope:
        weights = tf.Variable(tf.truncated_normal(
                  [BOTTLENECK_TENSOR_SIZE, 1024], stddev=0.001))
        biases = tf.get_variable(name='biases1', shape=[1024], 
                    initializer=tf.constant_initializer(0.1))
        fc1 = tf.nn.relu(tf.matmul(bottleneck_input, weights) + biases, name='fc1')
        tf.summary.scalar('fc1'+'/sparsity', tf.nn.zero_fraction(fc1))
        tf.summary.histogram('fc1'+'/activations', fc1)

    with tf.name_scope('fc2') as scope:
        weights = tf.Variable(tf.truncated_normal([1024, 512], stddev=0.001))
        biases = tf.get_variable(name='biases2', shape=[512],
                    initializer=tf.constant_initializer(0.1))
        fc2 = tf.nn.relu(tf.matmul(fc1, weights)+biases, name='fc2')
        tf.summary.scalar('fc2'+'/sparsity', tf.nn.zero_fraction(fc2))
        tf.summary.histogram('fc2'+'/activations', fc2)

    with tf.name_scope('softmax_linear') as scope:
        weights = tf.Variable(tf.truncated_normal([512, CLASSES], stddev=1/512.0))
        biases = tf.get_variable(name='biases3', shape=[CLASSES], 
                    initializer=tf.constant_initializer(0.0))
        logits = tf.matmul(fc2, weights)+biases
        final_tensor = tf.nn.softmax(logits)
        # tf.summary.scalar('softmax_linear'+'/sparsity', final_tensor)
        tf.summary.histogram('softmax_linear'+'/activations', final_tensor)

    return logits, final_tensor

"""
AM-Softmax
input:
    --embedding: 网络输出的logits归一化的值
    --label_batch: a batch of groundtruth
    --args：上一层网络大小
    --nrof_classes: 类别数量
return：
    --adjust_theta: 
"""
def AM_logits_compute(embeddings, label_onehot, args, nrof_classes):
    m = 0.35
    s = 30
    with tf.name_scope('AM_logits'):
        kernel = tf.get_variable(name='kernel', dtype=tf.float32,
                    shape=[args, nrof_classes],
                    initializer=tf.contrib.layers.xavier_initializer(uniform=False))
        kernel_norm = tf.nn.l2_normalize(kernel, 0, 1e-10, name='kernel_norm')
        cos_theta = tf.matmul(embeddings, kernel_norm)
        cos_theta = tf.clip_by_value(cos_theta, -1,1) # for numerical steady
        phi = cos_theta - m
        # label_onehot = tf.one_hot(label_batch, nrof_classes)
        adjust_theta = s * tf.where(tf.equal(label_onehot, 1), phi, cos_theta)

        return adjust_theta


def main(_):
    # 加载inception-v3
    with gfile.FastGFile(INCEPTION_DIR, 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
    # 获取数据输入层JPEG_DATA_TENSOR_NAME和瓶颈层BOTTLENECK_TENSOR_NAME
    bottleneck_tensor, jpeg_data_tensor = tf.import_graph_def(graph_def, 
        return_elements=[BOTTLENECK_TENSOR_NAME, JPEG_DATA_TENSOR_NAME])

    # 定义网络输入
    bottleneck_input = tf.placeholder(tf.float32, 
        [None, BOTTLENECK_TENSOR_SIZE], name='BottleneckInputPlaceholder')
    groundtruth_input = tf.placeholder(tf.float32,
        [None, CLASSES], name='GroundTruthPlaceholder')

    # 准确率率上不去了，一直在 5%-10%
    # logits, final_tensor = model(bottleneck_input)

    # 搭建一个全连接层进行分类
    with tf.name_scope('fc1'):
        weights = tf.Variable(tf.truncated_normal(
            [BOTTLENECK_TENSOR_SIZE, CLASSES], stddev=0.001))
        biases = tf.Variable(tf.zeros([CLASSES]))
        logits = tf.matmul(bottleneck_input, weights) + biases
        tf.summary.histogram('fc1'+'/pre_activation', logits)
        final_tensor = tf.nn.softmax(logits)
        tf.summary.histogram('fc1'+'/activation', final_tensor)

    # 增加AM-Softmax代替softmax
    # embeddings = tf.nn.l2_normalize(bottleneck_input, 1, 1e-10, name='embeddings')
    # AM_logits = AM_logits_compute(embeddings, groundtruth_input, 
    #                               BOTTLENECK_TENSOR_SIZE, CLASSES)

    # 定义交叉熵损失
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(
                            logits=logits, labels=groundtruth_input,
                            name='cross_entropy_per_example')
    cross_entropy_mean = tf.reduce_mean(cross_entropy)
    tf.summary.scalar('cross_entropy', cross_entropy_mean)
    # 指数衰减学习率
    global_step = tf.Variable(0, trainable=False)
    lr = tf.train.exponential_decay(learning_rate=LEARNING_RATE,
                                    global_step=global_step,
                                    decay_steps=DECAY_STEPS,
                                    decay_rate=DECAY_RATE,
                                    staircase=False)
    tf.summary.scalar('learning_rate', lr)

    # 优化
    train_step = tf.train.GradientDescentOptimizer(lr).minimize(
                    cross_entropy_mean)

    # 计算正确率
    with tf.name_scope('evaluation'):
        correct_prediction = tf.equal(tf.argmax(logits, 1), # final_tensor
                                tf.argmax(groundtruth_input, 1))
        evaluation_step = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        tf.summary.scalar('evaluation', evaluation_step)

    merged = tf.summary.merge_all()
    saver = tf.train.Saver(max_to_keep=1)
    max_acc = 0.7
    with tf.Session() as sess:
        writer = tf.summary.FileWriter(log_dir, tf.get_default_graph())
        tf.global_variables_initializer().run()

        for step in range(STEPS):
            # 每次获取一个batch的训练数据
            train_bottlenecks, train_groundtruth = get_batch_images(
                sess, JSON_TRAIN, TRAIN_IMAGES_DIR, 'train',
                jpeg_data_tensor, bottleneck_tensor)
            _, summary = sess.run([train_step, merged], 
                feed_dict={bottleneck_input: train_bottlenecks,
                           groundtruth_input: train_groundtruth})

            # 每100步输出一次正确率 在train中随机抽取 1 batch 的图像计算acc
            if step%100 == 0 or step + 1 == STEPS:
                cross_val_bottlenecks, cross_val_groundtruth = get_batch_images(
                    sess, JSON_TRAIN, TRAIN_IMAGES_DIR, 'train',
                    jpeg_data_tensor, bottleneck_tensor)
                cross_val_accuracy = sess.run(evaluation_step, 
                    feed_dict={bottleneck_input: cross_val_bottlenecks,
                               groundtruth_input: cross_val_groundtruth})
                print('Step %d: Accuracy on random sampled %d examples = %.3f%%' 
                    % (step, BATCH_SIZE, cross_val_accuracy*100))

                # 保存精度最高的模型
                if cross_val_accuracy > max_acc:
                    max_acc = cross_val_accuracy
                    saver.save(sess, './model/plant_disease.ckpt', global_step=step)

                writer.add_summary(summary, step)

            if step%1000 == 0 or step + 1 == STEPS:
                # 最后在val数据集中测试正确率
                val_bottlenecks, val_groundtruth = get_val_bottlenecks(sess,
                            JSON_VAL, VAL_IMAGES_DIR, jpeg_data_tensor, bottleneck_tensor)
                val_accuracy = sess.run(evaluation_step, feed_dict={
                    bottleneck_input: val_bottlenecks, groundtruth_input: val_groundtruth})
                print('Validation accuracy = %.3f%%' % (val_accuracy * 100))

        # 预测test中的数据，并保存成可提交的json格式
        test_bottlenecks, test_images = get_test_bottlenecks(sess,
                    TEST_IMAGES_DIR, jpeg_data_tensor, bottleneck_tensor)
        predict_test = sess.run(logits, # final_tensor
                        feed_dict={bottleneck_input: test_bottlenecks})
        predict_test = tf.argmax(predict_test, 1)

        # predict_test = np.squeeze(predict_test) 
        # bottleneck_string = ','.join(str(x) for x in predict_test)
        # print bottleneck_string

        result = []
        for index in range(len(test_images)):
            single = {}
            single["disease_class"] = int(predict_test[index].eval())
            single["image_id"] = test_images[index]
            result.append(single)
        # print result
        # with open('./test_result.txt', 'w') as f:
        #     f.write(result)
        # 写入json
        with open('./test_result.json', 'w') as f:
            f.write(json.dumps(result))

    writer.close()

if __name__ == '__main__':
    tf.app.run()
