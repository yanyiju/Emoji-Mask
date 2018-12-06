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
    data_as_array = np.zeros((len(expression_files), 100, 100))
    for i in range(len(expression_files)):
        file_num = int(expression_files[i][0:len(expression_files[i]) - 4])
        img = imageio.imread(expression_path + '/' + expression_files[i])
        img_resized = cv2.resize(img, (100, 100))
        img_resized_grey = color.rgb2gray(img_resized)
        data_as_array[file_num - 1] = img_resized_grey
    return data_as_array


'''WRITE ALL FUNCTIONS HERE'''


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
        plt.imsave('out/' + str(i) + ".png", pca[:, i].reshape([28, 28]))
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
    # expression_train_labels = read_label(label_path)

    '''CALL YOUR FUNCTIONS HERE.

    Please call of your functions here. For this problem, we ask you to
    visualize several things. You need to do this yourself (in any
    way you wish).

    For the MNIST and SVHN error rates, please store these two error
    rates in the variables called "mnist_error_rate" and
    "svhn_error_rate", for the MNIST error rate and SVHN error rate,
    respectively. These two variables will be used to write
    the numbers in a JSON file.
    '''

    # setting parameters
    pca_dimension = 10
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

    mnist_error_count = 0
    mnist_test_num = 500
    mnist_failed_instances = []
    mnist_correct_example_saved = False
    mnist_wrong_example_saved = False
    for i in range(mnist_test_num):
        neighbors = get_neighbors(train_img_projection, mnist_test_images[i], k_neighbors, pca_components, mean)
        response = get_response(neighbors, mnist_train_labels)
        if response != mnist_test_labels[i]:
            mnist_error_count += 1
            mnist_failed_instances.append(i)
            if not mnist_wrong_example_saved:
                mnist_wrong_example_saved = True
                plt.imsave('out/mnist_wrong_response_' + str(response) +
                           '_correct_' + str(mnist_test_labels[i]) + '.png', mnist_test_images[i])
        else:
            if not mnist_correct_example_saved:
                mnist_correct_example_saved = True
                plt.imsave('out/mnist_correct_' + str(mnist_test_labels[i]) + '.png', mnist_test_images[i])

    svhn_error_count = 0
    svhn_digit_count = 0
    svhn_failed_instances = []
    for i in range(300):
        image = etai.read(base_svhn_path + '/' + digits[i]['filename'])
        image = etai.rgb_to_gray(image)
        for b in digits[i]['boxes']:
            svhn_digit_count += 1
            chopped = image[int(float(b['top'])):int(float(b['top']) + float(b['height'])),
                      int(float(b['left'])): int(float(b['left']) + float(b['width']))]
            chopped_resized = etai.resize(chopped, 28, 28)
            neighbors = get_neighbors(train_img_projection, chopped_resized, k_neighbors, pca_components, mean)
            response = get_response(neighbors, mnist_train_labels)
            if response != b['label']:
                svhn_error_count += 1
                svhn_failed_instances.append((digits[i]['filename'], b, response))
    print(svhn_failed_instances[0])


    # Make sure you assign values to these two variables
    mnist_error_rate = mnist_error_count / mnist_test_num
    svhn_error_rate = svhn_error_count / svhn_digit_count

    error_rate_dic = defaultdict(lambda: defaultdict())
    error_rate_dic["error_rates"]["mnist_error_rate"] = mnist_error_rate
    error_rate_dic["error_rates"]["svhn_error_rate"] = svhn_error_rate
    etas.write_json(error_rate_dic, data.error_rate_file)


if __name__ == "__main__":
    run("../expression_set_img", "../expression_set_label")
