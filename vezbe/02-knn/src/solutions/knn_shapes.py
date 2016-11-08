# author: Aleksandar Novakovic

from skimage.io import imread
from skimage.color import rgb2gray
from skimage.measure import label, regionprops
from skimage.morphology import erosion, disk
from scipy import ndimage as ndi
import numpy as np
from knn_classifier import KNNClassifier
import csv
import math


def get_properties(region):
    bbox = region.bbox
    height = bbox[2] - bbox[0]
    width = bbox[3] - bbox[1]
    radius = (width + height) / (2. * region.equivalent_diameter)
    perimeter = region.perimeter / (region.equivalent_diameter * math.pi)
    return [radius, perimeter]


def get_elements_from_image(img_path):
    img = 1. - rgb2gray(imread(img_path))
    img = ndi.binary_fill_holes((img > 0)).astype('float64')
    img = erosion(img, selem=disk(6))
    _labels = label(img)
    regions = regionprops(_labels)
    regions = sorted(regions, key=lambda elem: elem.bbox[1])
    ret_val = []
    for region in regions:
        ret_val.append(get_properties(region))
    return ret_val


def draw_regions(regs, img_size):
    img = np.ndarray((img_size[0], img_size[1]), dtype='float64')
    for reg in regs:
        coords = reg.coords
        for coord in coords:
            img[coord[0]][coord[1]] = 1.
    return img


if __name__ == '__main__':

    train_data_path = './../../data/train/'
    test_data_path = './../../data/test/'
    all_labels = {}
    data = []
    data_labels = []

    with open(train_data_path + 'labels.txt', 'r') as f:
        content = csv.DictReader(f)
        for lbl in content:
            all_labels[lbl['class_label']] = lbl['class_name']

    for lbl in all_labels.keys():
        group_data = get_elements_from_image(train_data_path + all_labels[lbl] + '.png')
        data += group_data
        data_labels += [lbl] * len(group_data)

    classifier = KNNClassifier(1, distance='L1')
    classifier.fit(data, data_labels)
    test_data = get_elements_from_image(test_data_path + 'shapes.png')
    test_results = []
    with open(test_data_path + 'labels.txt', 'r') as f:
        rows = csv.reader(f)
        expected = rows.next()
    results = classifier.predict(test_data)
    acc = 0.
    for r, e in zip(results, expected):
        acc += 1 if r == e else 0
    acc /= len(results)
    print results
    print 'Accuracy: %.2f %%' % (acc * 100)
