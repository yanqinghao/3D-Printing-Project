"""
功能：实现对指定大小的生成图片进行sample与label分类制作
def get_file(file_dir)

"""

import os
import numpy as np
from PIL import Image
import tensorflow as tf
from suanpan.log import logger


def get_file(file_dir, label_map):
    # step1：获取路径下所有的图片路径名，存放到
    # 对应的列表中，同时贴上标签，存放到label列表中

    images = {}
    label2num = {}
    for name, num in label_map.items():
        images[name] = []
        label2num[name] = []

    for label, num in label_map.items():
        for file in os.listdir(os.path.join(file_dir, label)):
            images[label].append(os.path.join(file_dir, label, file))
            label2num[label].append(num)

    # 打印出提取图片的情况，检测是否正确提取
    for label, num in label_map.items():
        logger.info("There are {} {}".format(len(images[label]), label))

    # step2：对生成的图片路径和标签List做打乱处理把所有的合起来组成一个list（img和lab）
    # 合并数据numpy.hstack(tup)
    # tup可以是python中的元组（tuple）、列表（list），或者numpy中数组（array），函数作用是将tup在水平方向上（按列顺序）合并
    image_list = np.hstack((*list(images.values()),))
    label_list = np.hstack((*list(label2num.values()),))

    return image_list, label_list


# 将image和label转为list格式数据，因为后边用到的的一些tensorflow函数接收的是list格式数据
# 为了方便网络的训练，输入数据进行batch处理
# image_W, image_H, ：图像高度和宽度
# batch_size：每个batch要放多少张图片
# capacity：一个队列最大多少
def get_batch(image, label, image_W, image_H, batch_size, capacity):
    # step1：将上面生成的List传入get_batch() ，转换类型，产生一个输入队列queue
    # tf.cast()用来做类型转换
    image = tf.cast(image, tf.string)  # 可变长度的字节数组.每一个张量元素都是一个字节数组
    label = tf.cast(label, tf.int32)
    # tf.train.slice_input_producer是一个tensor生成器
    # 作用是按照设定，每次从一个tensor列表中按顺序或者随机抽取出一个tensor放入文件名队列。
    input_queue = tf.train.slice_input_producer([image, label])
    label = input_queue[1]
    image_contents = tf.read_file(input_queue[0])  # tf.read_file()从队列中读取图像

    # step2：将图像解码，使用相同类型的图像
    image = tf.image.decode_jpeg(image_contents, channels=3)
    # jpeg或者jpg格式都用decode_jpeg函数，其他格式可以去查看官方文档

    # step3：数据预处理，对图像进行旋转、缩放、裁剪、归一化等操作，让计算出的模型更健壮。
    image = tf.image.resize_image_with_crop_or_pad(image, image_W, image_H)
    # 对resize后的图片进行标准化处理
    image = tf.image.per_image_standardization(image)

    # step4：生成batch
    # image_batch: 4D tensor [batch_size, width, height, 3], dtype = tf.float32
    # label_batch: 1D tensor [batch_size], dtype = tf.int32
    image_batch, label_batch = tf.train.shuffle_batch(
        [image, label],
        batch_size=batch_size,
        num_threads=16,
        capacity=capacity,
        min_after_dequeue=32,
    )

    # 重新排列label，行数为[batch_size]
    label_batch = tf.reshape(label_batch, [batch_size])
    image_batch = tf.cast(image_batch, tf.float32)  # 显示灰度图

    return image_batch, label_batch


def get_image(image_list):
    for image_path in image_list:
        image = Image.open(image_path)
        image = image.resize([72, 72])
        image_arr = np.array(image)
        image = tf.cast(image_arr, tf.float32)
        # print('1', np.array(image).shape)
        image = tf.image.per_image_standardization(image)
        # print('2', np.array(image).shape)
        image = tf.reshape(image, [1, 72, 72, 3])
        yield image


def get_label(label_list):
    for label in label_list:
        yield label


def get_predict_file(file_dir):
    images = []
    for file in os.listdir(os.path.join(file_dir)):
        images.append(os.path.join(file_dir, file))
    image_list = np.hstack((*images,))

    return image_list


def get_od_image(image_list):
    for image_path in image_list:
        image = Image.open(image_path)
        (im_width, im_height) = image.size
        yield np.array(image.getdata()).reshape((im_height, im_width, 3)).astype(
            np.uint8
        )
