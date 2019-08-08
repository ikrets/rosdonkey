import cv2
import numpy as np
from glob import glob
import tensorflow as tf
import os
from PIL import Image
from sklearn.model_selection import train_test_split
from io import BytesIO

from transform import make_undistort_birdeye

def prepare_file_dataset(folders, dataset_name, transformation=lambda x: x):
    images = []
    for folder in folders:
        images.extend(glob(os.path.join(folder, 'images/*.jpg')))

    masks = [i.replace('images', 'masks').replace('jpg', 'png') for i in images]
    os.makedirs(dataset_name, exist_ok=True)

    for i in range(len(images)):
        folder, name = os.path.split(images[i])
        name = os.path.splitext(name)[0]

        img = np.array(Image.open(images[i]))
        img[:, -2:, :] = 0

        img = transformation(img)
        img = Image.fromarray(img)
        img.save(os.path.join(dataset_name, name + '_img.png'))

        mask = np.array(Image.open(masks[i]))
        mask[:, -2:] = 0
        mask = transformation(mask)
        mask = Image.fromarray(mask)
        mask.save(os.path.join(dataset_name, name + '_mask.png'))

def prepare_tfrecord_dataset(folder, dataset_name, transformation=lambda x: x):
    images = sorted(glob(os.path.join(folder, 'images/*.jpg')))
    masks = [i.replace('images', 'masks').replace('jpg', 'png') for i in images]

    examples_train_val = []
    examples_train_only = []

    keywords_val = ['home']

    for i in range(len(images)):
        img = np.array(Image.open(images[i]))
        img[:, -2:, :] = 0

        img = transformation(img)
        img = Image.fromarray(img)
        img_bytes = BytesIO()
        img.save(img_bytes, format='PNG')

        img_feature = tf.train.BytesList(value=[img_bytes.getvalue()])
        img_feature = tf.train.Feature(bytes_list=img_feature)

        mask = np.array(Image.open(masks[i]))
        mask[:, -2:] = 0
        mask = transformation(mask)
        mask = Image.fromarray(mask)
        mask_bytes = BytesIO()
        mask.save(mask_bytes, format='PNG')

        mask_feature = tf.train.BytesList(value=[mask_bytes.getvalue()])
        mask_feature = tf.train.Feature(bytes_list=mask_feature)

        name = os.path.splitext(os.path.split(images[i])[-1])[0]

        features = tf.train.Features(feature={
            'image': img_feature,
            'name': tf.train.Feature(
                bytes_list=tf.train.BytesList(value=[name.encode('utf-8')])),
            'width': tf.train.Feature(int64_list=tf.train.Int64List(value=[img.width])),
            'height': tf.train.Feature(int64_list=tf.train.Int64List(value=[img.height])),
            'mask': mask_feature
        })

        val = False
        for k in keywords_val:
            if k in name:
                examples_train_val.append(tf.train.Example(features=features))
                val = True
                break

        if not val:
            examples_train_only.append(tf.train.Example(features=features))

    train, val = train_test_split(examples_train_val, test_size=0.2)

    with tf.python_io.TFRecordWriter(dataset_name + '_train.tfrecord') as tfwriter:
        for example in train + examples_train_only:
            tfwriter.write(example.SerializeToString())

    with tf.python_io.TFRecordWriter(dataset_name + '_val.tfrecord') as tfwriter:
        for example in val:
            tfwriter.write(example.SerializeToString())

if __name__ == '__main__':
    def expand(img, target_height):
        if len(img.shape) == 2:
            img = img[:, :, np.newaxis]

        expanded = np.zeros((target_height, img.shape[1], img.shape[2]), dtype=img.dtype)
        expanded[-img.shape[0]:, :, :] = img

        return np.squeeze(expanded)

    def resize(img):
        resized = cv2.resize(img, (320, 240), cv2.INTER_LINEAR)
        return resized

    undistort_birdeyeview = make_undistort_birdeye(input_shape=(320, 240), target_shape=(32, 48))

    prepare_file_dataset(['/home/ilya/random steering/datasets/extracted',
                          '/home/ilya/random steering/datasets/extracted-sayat'],
                         '/home/ilya/random steering/datasets/transformed-resized320x240-to32x48',
                         lambda img: undistort_birdeyeview(resize(expand(img, target_height=480))))
