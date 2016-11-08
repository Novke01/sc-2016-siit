# author: Aleksandar Novakovic

import sys
import operator as oper

class KNNClassifier(object):

    def __init__(self, k, distance='L2'):
        """
        Konstruktor KNN klasifikatora.

        :param k: koliko najblizih suseda uzeti u obzir.
        :param distance: koju udaljenost koristiti za poredjenje, moguce vrednosti su 'L1' i 'L2'
        """

        # validna vrednost parametra *k* je ceo broj, veci od 0
        # validna vrednost parametra *distance* je 'L1' ili 'L2'
        self.k = k
        if distance == 'L1':
            self.distance = KNNClassifier._euclidean_distance
        elif distance == 'L2':
            self.distance = KNNClassifier._manhattan_distance
        self.data = []

    def fit(self, X, y):
        """
        :param X: podaci
        :param y: labele klasa (celobrojne vrednosti >= 0)
        :return: None
        """
        # prikazati poruku upozorenja ako je k jednako broju klasa
        labels_num = len(y)
        if labels_num == self.k:
            print 'Warning: k is equal to number of labels.'

        for point, label in zip(X, y):
            self.data.append([point, label])

    def predict(self, X):
        """
        :param X: podaci za klasifikaciju
        :return: labele X podataka nakon klasifikacije
        """
        # (X je lista podataka, dakle nije nuzno samo jedan podatak)
        # povratna vrednost su odgovarajuce labele
        ret_val = []
        for x in X:
            distances = []
            for d in self.data:
                distances.append([self.distance(x, d[0]), d[1]])
            distances.sort(key=lambda point: point[0])
            labels = {}
            for i in xrange(self.k):
                distance = distances[i]
                if distance[1] in labels:
                    labels[distance[1]] += 1
                else:
                    labels[distance[1]] = 1
            ret_val.append(max(labels.iteritems(), key=oper.itemgetter(1))[0])
        return ret_val

    @staticmethod
    def _manhattan_distance(a, b):
        result = 0
        for coord_a, coord_b in zip(a, b):
            result += abs(coord_a - coord_b)
        return result

    @staticmethod
    def _euclidean_distance(a, b):
        result = 0
        for coord_a, coord_b in zip(a, b):
            diff = coord_a - coord_b
            result += diff * diff
        return result
