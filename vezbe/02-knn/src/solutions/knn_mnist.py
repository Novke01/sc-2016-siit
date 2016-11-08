# author: Aleksandar Novakovic

from sklearn.datasets import fetch_mldata
import matplotlib.pyplot as plt
from skimage.measure import label, regionprops
import numpy as np
import random as rnd
import math

from knn_classifier import KNNClassifier


def get_properties(region):
    bbox = region.bbox
    height = bbox[2] - bbox[0]
    width = bbox[3] - bbox[1]
    bbox_area = width * height
    width_height_prop = float(width) / height
    area_prop = float(region.filled_area) / region.convex_area
    perimeter = region.perimeter / (region.equivalent_diameter * math.pi)
    return [width_height_prop, area_prop, perimeter]


if __name__ == '__main__':

    mnist = fetch_mldata('MNIST original')

    combined = list(zip(mnist.data, mnist.target))
    rnd.shuffle(combined)

    mnist.data[:], mnist.target[:] = zip(*combined)

    regions = []

    for elem in mnist.data:
        elem = elem.tolist()
        regions.append(elem)
    #     elem = (elem / 255.).astype('float64')
    #     elem.shape = (28L, 28L)
    #     elem = elem > 0.
    #
    #     labels = label(elem)
    #     regions.append(regionprops(labels)[0])

    n = len(mnist.data)
    train_len = int(n * .8)
    test_len = n - train_len

    train_labels = mnist.target[:train_len].astype('uint8')
    test_labels = mnist.target[train_len: (train_len + 50)].astype('uint8')

    print 'start training...'

    # train_data = []
    # for train_index in xrange(train_len):
    #     train_data.append(get_properties(regions[train_index]))

    classifier = KNNClassifier(25, distance='L1')
    classifier.fit(regions[:train_len], train_labels)

    print 'stop training...'
    print 'start testing...'

    # test_data = []
    # for test_index in xrange(train_len, n):
    #     test_data.append(get_properties(regions[test_index]))

    results = classifier.predict(regions[train_len:(train_len + 50)])

    print 'stop testing...'
    count = 0
    for test_label, result in zip(test_labels, results):
        if test_label == result:
            count += 1

    acc = float(count) / 50
    print 'Accuracy: %.2f %%' % (acc * 100)
