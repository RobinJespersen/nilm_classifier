import numpy
from sklearn.base import ClassifierMixin, BaseEstimator
from sklearn.neighbors import KNeighborsClassifier
from collections import Counter


class Location(object):
    """
    Represents a location that is working on its own, with private data, and can be queried from the master
    """

    def __init__(self, name, n_neighbors):
        self.n_neighbors = n_neighbors
        self.name = name
        self.clf = KNeighborsClassifier(algorithm='auto', n_neighbors=self.n_neighbors)
        self.X = []
        self.y = []
        self.popularity = {}

    def add_data(self, X, y):
        self.X.append(X)
        self.y.append(y)
        if y not in self.popularity.keys():
            self.popularity[y] = 0
        self.popularity[y] += 1

    def fit(self):
        self.clf.fit(self.X, self.y)

    def predict(self, X):
        result = []

        predictions_with_neighbors: tuple = self.clf.kneighbors(X, n_neighbors=self.n_neighbors, return_distance=True)
        for i in range(len(X)):
            for j in range(self.n_neighbors):
                label = self.y[predictions_with_neighbors[1][i][j]]
                distance = predictions_with_neighbors[0][i][j]
                imbalance = (self.popularity[label] / len(self.X))
                result.append((label, distance, imbalance))

        return result


class WfdknnClassifier(ClassifierMixin, BaseEstimator):

    def __init__(self, location_column_index=0, n_neighbors=1):
        self.location_column_index = {}
        self.locations = {}

        """
        :param location_column_index: index of the column containing the location in X, is deleted
        before being passed down to the location
        """
        self.location_column_index = location_column_index

        """
        :param n_neighbors: int
        """
        self.n_neighbors = n_neighbors

    def fit(self, X, y):
        self.locations = {}

        for i in range(len(X)):
            xi = X[i]
            yi = y[i]
            location = xi[self.location_column_index]
            xi = numpy.delete(xi, self.location_column_index, 0)

            if location not in self.locations.keys():
                self.locations[location] = Location(location, n_neighbors=self.n_neighbors)

            self.locations[location].add_data(xi, yi)

        for location in self.locations:
            self.locations[location].fit()

    def predict(self, X):
        """Predict the class labels for the provided data.

        Parameters
        ----------
        X : array-like of shape (n_queries, n_features), \
                or (n_queries, n_indexed) if metric == 'precomputed'
            Test samples.

        Returns
        -------
        y : ndarray of shape (n_queries,) or (n_queries, n_outputs)
            Class labels for each data sample.
        """
        X = numpy.delete(X, self.location_column_index, 1)

        predictions = {}
        for location in self.locations.keys():
            predictions[location] = self.locations[location].predict(X)

        result = []
        for i in range(len(X)):
            r = {}
            for location in self.locations.keys():
                (label, _, imbalance) = predictions[location][i]
                if label not in r.keys():
                    r[label] = 0
                r[label] += (1-imbalance)

            winner_label = None
            winner_value = 0
            for k in r:
                if r[k] > winner_value:
                    winner_value = r[k]
                    winner_label = k
            result.append(winner_label)

        return result
