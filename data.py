import os
import cv2
import random
import numpy as np
import tensorflow as tf
from tensorflow.python.data.ops.dataset_ops import AUTOTUNE


def shuffle(images, labels):
    if images is None or len(images) == 0:
        return [], []
    tmp = list(zip(images, labels))
    random.shuffle(tmp)
    images, labels = zip(*tmp)
    return list(images), list(labels)


def load_dataset(dataset_dir, batch_size, valid_ratio=0.2, one_hot=True,
                 do_shuffle=True, normalize=True, resize=None):
    train_images = []
    train_labels = []
    valid_images = []
    valid_labels = []
    classes = 0
    for lbl_dir in os.listdir(dataset_dir):
        if os.path.isdir(os.path.join(dataset_dir, lbl_dir)):
            classes = classes + 1
            images = []
            labels = []
            for f in os.listdir(os.path.join(dataset_dir, lbl_dir)):
                if not f.endswith(('.png', '.jpeg', 'jpg', '.bmp')):
                    continue
                image_filepath = os.path.join(dataset_dir, os.path.join(lbl_dir, f))
                image = cv2.imread(image_filepath, cv2.IMREAD_COLOR).astype(np.float32)
                if resize is not None:
                    image = cv2.resize(image, resize)
                images.append(image)
                labels.append(int(lbl_dir.split(".")[0]))
            if do_shuffle:
                images, labels = shuffle(images, labels)
            img_num = len(images)
            valid_idx = int(img_num * valid_ratio)
            train_images.extend(images[valid_idx:])
            train_labels.extend(labels[valid_idx:])
            valid_images.extend(images[:valid_idx])
            valid_labels.extend(labels[:valid_idx])

    if do_shuffle:
        train_images, train_labels = shuffle(train_images, train_labels)
        valid_images, valid_labels = shuffle(valid_images, valid_labels)

    train_images = np.array(train_images, dtype=np.float32)
    train_labels = np.array(train_labels, dtype=np.int32)
    valid_images = np.array(valid_images, dtype=np.float32)
    valid_labels = np.array(valid_labels, dtype=np.int32)

    if normalize:
        train_images = train_images / 255.
        valid_images = valid_images / 255.

    img_num = train_images.shape[0]
    val_img_num = valid_images.shape[0]
    train_steps = int(img_num / batch_size) + bool(img_num % batch_size)
    valid_steps = int(val_img_num / batch_size) + bool(val_img_num % batch_size)

    train_images = tf.data.Dataset.from_tensor_slices(train_images)
    train_labels = tf.data.Dataset.from_tensor_slices(train_labels)
    valid_images = tf.data.Dataset.from_tensor_slices(valid_images)
    valid_labels = tf.data.Dataset.from_tensor_slices(valid_labels)
    if one_hot:
        train_labels = train_labels.map(lambda x: make_one_hot(x, tf.constant(classes, dtype=tf.int32)),
                                        num_parallel_calls=AUTOTUNE)
        valid_labels = valid_labels.map(lambda x: make_one_hot(x, tf.constant(classes, dtype=tf.int32)),
                                        num_parallel_calls=AUTOTUNE)

    train_dataset = tf.data.Dataset.zip((train_images, train_labels))
    valid_dataset = tf.data.Dataset.zip((valid_images, valid_labels))

    train_dataset = train_dataset.batch(batch_size).prefetch(AUTOTUNE)
    valid_dataset = valid_dataset.batch(batch_size).prefetch(AUTOTUNE)

    return train_dataset, valid_dataset, train_steps, valid_steps, classes


def make_one_hot(lbls, classes):
    labels = tf.one_hot(lbls, classes, dtype=tf.float32)
    return labels
