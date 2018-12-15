#!/usr/bin/env python
'''
A module for classifying the SVHN (Street View House Number) dataset
using an eigenbasis.

Info:
    type: eta.core.types.Module
    version: 0.1.0
'''
# pragma pylint: disable=redefined-builtin
# pragma pylint: disable=unused-wildcard-import
# pragma pylint: disable=wildcard-import
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
from builtins import *
from collections import defaultdict
# pragma pylint: enable=redefined-builtin
# pragma pylint: enable=unused-wildcard-import
# pragma pylint: enable=wildcard-import

import cv2
import gzip
import matplotlib.pyplot as plt
import imageio
import numpy as np
import os
from skimage import color
import struct
import sys

img_row = 50
img_col = 50

emotion_labels = [
    'NE',
    'AN',
    'contempt',
    'DI',
    'FE',
    'HA',
    'SA',
    'SU'
]


def read_expression(expression_path):
    '''Reads the expressions in as an array.

    Args:
        expression_path: the path of the expression path

    Returns:
        data_as_array: a numpy array corresponding to the data within the
            expression path. The output is a
            (n, 28, 28) numpy array, where n is the number of images.
    '''
    expression_files = os.listdir(expression_path)
    data_as_array = np.zeros((len(expression_files), img_row, img_col))
    for i in range(len(expression_files)):
        file_num = int(expression_files[i][0:len(expression_files[i]) - 4])
        img = imageio.imread(expression_path + '/' + expression_files[i])
        img_resized = cv2.resize(img, (img_row, img_col))
        img_resized_grey = color.rgb2gray(img_resized)
        data_as_array[file_num - 1] = img_resized_grey
    return data_as_array


def read_expression_idx(expression_idx_path):
    '''Reads the emotion labels in as an array.'''
    expression_label_file = os.listdir(expression_idx_path)
    labels = []
    for i in range(len(expression_label_file)):
        labels.append(0)
    for i in range(len(expression_label_file)):
        file_num = int(expression_label_file[i][0:len(expression_label_file[i]) - 4])
        f = open(expression_idx_path + '/' + str(file_num) + ".txt", "r")
        line = f.readline()
        labels[file_num - 1] = int(float(line))
    return labels


def get_pca(train_img, pca_dimension):
    flattened_img = np.zeros((train_img.shape[1] * train_img.shape[2], train_img.shape[0]))
    for i in range(len(train_img)):
        flattened_img[:, i] = np.transpose(train_img[i].flatten())
    pca = np.zeros((flattened_img.shape[0], pca_dimension))
    cov = np.cov(flattened_img)
    eigenvalues, eigenvectors = np.linalg.eig(cov)
    top_eig_idx = sorted(range(len(eigenvalues)), key=lambda i: eigenvalues[i])[-pca_dimension:]
    for i in range(pca_dimension):
        pca[:, i] = eigenvectors[:, top_eig_idx[i]]
        # etai.write(255 * pca[:, i].reshape([28, 28]), str(i) + ".png")
        plt.imsave('out/' + str(i) + ".png", pca[:, i].reshape([img_row, img_col]))
        # plt.show()
    return pca


def get_item(item):
    return item[1]


def get_neighbors(train, test_instance, num, pca_components, mean):
    distance = []
    neighbors = []
    test_projection = np.zeros(pca_components.shape[1])
    for j in range(pca_components.shape[1]):
        test_projection[j] = np.dot(test_instance.flatten() - mean,
                                    np.transpose(pca_components[:, j])) / np.linalg.norm(
            pca_components[:, j])
    for i in range(len(train)):
        distance.append((i, np.linalg.norm(train[i] - test_projection)))
    distance.sort(key=get_item)
    for i in range(num):
        neighbors.append(distance[i][0])
    return neighbors


def get_response(neighbors, labels):
    poll = np.zeros(10)
    for n in neighbors:
        poll[labels[n]] += 1
    response = sorted(range(len(poll)), key=lambda i: poll[i])[-1:]
    return response


def run(expression_path, label_path):

    expression_train_images = read_expression(expression_path)
    expression_train_labels = read_expression_idx(label_path)

    # setting parameters
    pca_dimension = 9
    k_neighbors = 10
    pca_components = get_pca(expression_train_images, pca_dimension)
    flattened_img = np.zeros(
        (len(expression_train_images), expression_train_images[0].shape[0] * expression_train_images[0].shape[1]))
    for i in range(len(expression_train_images)):
        flattened_img[i] = expression_train_images[i].flatten()
    mean = np.mean(flattened_img, axis=0)
    train_img_projection = np.zeros((len(expression_train_images), pca_dimension))
    for i in range(len(expression_train_images)):
        for j in range(pca_dimension):
            train_img_projection[i, j] = np.dot(expression_train_images[i].flatten() - mean,
                                                np.transpose(pca_components[:, j])) / np.linalg.norm(
                pca_components[:, j])

    # test_path = "../face_detection/detected_faces"
    test_path = "../jaffe"
    test_file = os.listdir(test_path)
    test_data = np.zeros((len(test_file), img_row, img_col))
    correct_count = 0
    error_count = 0
    for i in range(len(test_file)):
        img = imageio.imread(test_path + '/' + test_file[i])
        img_resized = cv2.resize(img, (img_row, img_col))
        img_resized_grey = color.rgb2gray(img_resized)
        neighbors = get_neighbors(train_img_projection, img_resized_grey, k_neighbors, pca_components, mean)
        response = get_response(neighbors, expression_train_labels)
        if emotion_labels[response[0]] == test_file[i][3:5]:
            correct_count += 1
        else:
            error_count += 1
    print(correct_count / (correct_count + error_count))



if __name__ == "__main__":
    run("../expression_set_img", "../expression_set_label")
