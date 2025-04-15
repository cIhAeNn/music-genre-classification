from python_speech_features import mfcc
import scipy.io.wavfile as wav
import numpy as np

from tempfile import TemporaryFile
import os
import pickle
import random
import operator
import math
from collections import defaultdict

def distance(instance1, instance2, k):
    mm1, cm1 = instance1[0], instance1[1]
    mm2, cm2 = instance2[0], instance2[1]
    dist = np.trace(np.dot(np.linalg.inv(cm2), cm1))
    # np.linalg.inv(cm2):
    # This computes the inverse of the covariance matrix from instance2.
    # In many distance measures (like the Mahalanobis distance), the covariance (or its inverse) is used to scale differences according to how much the data vary in each direction.

    # np.dot(np.linalg.inv(cm2), cm1):
    # Multiplying the inverse of cm2 with cm1 gives a matrix that, roughly speaking, measures how the spread (dispersion) of instance1 differs relative to that of instance2.

    # np.trace(...):
    # The trace operation sums the diagonal elements of a square matrix. Here it aggregates the differences across all feature dimensions—producing a single scalar value that quantifies the overall difference between the covariance structures of the two instances.

    # Interpretation:
    # A larger value means that the way features vary (their dispersion) in instance1 is quite different when measured in the “units” defined by instance2’s variability.
    
    dist += np.trace(np.dot((mm2 - mm1).transpose(), np.linalg.inv(cm2)), (mm2 - mm1))
    # (mm2 - mm1):
    # This vector represents the difference between the mean feature values of instance2 and instance1.

    # np.linalg.inv(cm2):
    # Again, the inverse of instance2’s covariance is used to “normalize” the mean differences based on the spread of data in instance2.
    # Regions with low variance are more significant—small differences there are “penalized” more.

    # np.dot(np.dot((mm2 - mm1).transpose(), np.linalg.inv(cm2)), (mm2 - mm1)):
    # This double dot product effectively computes the squared Mahalanobis distance between the two mean vectors.
    # Unlike the Euclidean distance, the Mahalanobis distance takes the underlying variance (and correlations) into account.
    # This tells you how “surprising” or significant the difference in means is given the expected variability in instance2.

    # Interpretation:
    # A higher value indicates that the average spectral content (as captured by the mean MFCCs) of the two audio files differs greatly once you adjust for the typical variability in instance2.
    
    dist += np.log(np.linalg.det(cm2)) - np.log(np.linalg.det(cm1))
    # np.linalg.det(cm2) and np.linalg.det(cm1):
    # The determinants of the covariance matrices give a measure of the “volume” or overall spread of the data represented by each covariance matrix.

    # np.log(...):
    # Taking the logarithm turns the volume ratios into differences that are easier to add to the other terms.

    # Difference of log-determinants:
    # The term np.log(np.linalg.det(cm2)) - np.log(np.linalg.det(cm1)) measures the relative size of the covariance (spread) between instance2 and instance1.
    # This term adjusts the distance based on how “uncertain” each set of features is.
    
    dist -= k
    # Why subtract k?
    # In the standard formula for the Kullback–Leibler divergence between two Gaussian distributions, one term is the subtraction of the dimensionality of the feature space. Here, the parameter k may be used to incorporate that correction (or another constant offset chosen experimentally) to calibrate the final distance metric.

    return dist


def getNeighbors(trainingSet, instance, k):
    distances = []
    
    for x in range(len(trainingSet)):
        dist = distance(trainingSet[x], instance, k) + distance(instance, trainingSet[x], k)
        distances.append(trainingSet[x][2], dist)
    
    distances.sort(key=operator.itemgetter(1))
    neighbors=[distances[x][0] for x in range(k)]
    
    return neighbors