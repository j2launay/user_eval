#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from random import random
from scipy.stats import chi2
import scipy as sp
import pandas as pd
import math

def distances(x, y, ape, metrics='mahalanobis', dataset=None):
    if metrics == 'mahalanobis':
        distance = 0
        if ape.max_mahalanobis == None:
            df_x = pd.DataFrame(x, columns=ape.feature_names)
            df = pd.DataFrame(x, columns=ape.feature_names)
            df_x['mahalanobis'] = mahalanobis(x=df_x, data=df[ape.feature_names], max_mahalanobis=ape.max_mahalanobis)
            df_x['p'] = 1 - chi2.cdf(df_x['mahalanobis'], 3)
            return df_x
        else:
            if dataset is not None:
                df = pd.DataFrame(y, columns=ape.feature_names)
            else:
                df = pd.DataFrame(ape.test_data, columns=ape.feature_names)
            df_x = pd.DataFrame([x], columns=ape.feature_names)
            df_x['mahalanobis'] = mahalanobis(x=df_x, data=df[ape.feature_names], max_mahalanobis=ape.max_mahalanobis)
            df_x['p'] = 1 - chi2.cdf(df_x['mahalanobis'], 3)
            return df_x['mahalanobis'].item()
        #calculate p-value for each mahalanobis distance 
        
    continuous_features = [x for x in set(range(len(x))).difference(ape.categorical_features)]
    x_categorical, y_categorical = x[ape.categorical_features], y[ape.categorical_features]
    x_continuous, y_continuous = x[continuous_features], y[continuous_features]
    # We divide by 2 since the categorical data must have been one hot encoded before
    same_coordinates_categorical = (x_categorical.shape[0] - sum(x_categorical == y_categorical)) /2
    
    distance = 0
    for nb_feature in range(x_continuous.shape[0]):
        distance += abs(x_continuous[nb_feature] - y_continuous[nb_feature])/(ape.max_features[nb_feature] - ape.min_features[nb_feature])
    distance = distance + same_coordinates_categorical
    distance = distance/x.shape[0]

    
    distance = 0
    for nb_feature in range(x_continuous.shape[0]):
        temp_distance = ((x_continuous[nb_feature] - ape.mean_features[nb_feature]) - (y_continuous[nb_feature] - ape.mean_features[nb_feature]))\
            / ape.feature_variance[continuous_features[nb_feature]]
        distance += temp_distance * temp_distance
    distance = math.sqrt(distance)
    distance += same_coordinates_categorical
    try:
        distance = distance/ape.farthest_distance
    except AttributeError:
        # la distance au contrefactuelle le plus loin n'est pas encore calculé
        pass
    except TypeError:
        #On passe a une nouvelle instance et la distance max doit être mesurée de nouveau
        pass
    except ZeroDivisionError:
        pass
    
    return distance

def mahalanobis(x=None, data=None, cov=None, max_mahalanobis=None):
    """Compute the Mahalanobis Distance between each row of x and the data  
    x    : vector or matrix of data with, say, p columns.
    data : ndarray of the distribution from which Mahalanobis distance of each observation of x is to be computed.
    cov  : covariance matrix (p x p) of the distribution. If None, will be computed from data.
    """
    x_minus_mu = x - np.mean(data)
    if not cov:
        cov = np.cov(data.values.T)
    inv_covmat = sp.linalg.inv(cov)
    left_term = np.dot(x_minus_mu, inv_covmat)
    mahal = np.dot(left_term, x_minus_mu.T)
    if max_mahalanobis != None:
        mahal /= max_mahalanobis
    return mahal.diagonal()

def generate_inside_field(center, segment, n, max_features, min_features, feature_variance):
    """
    Args:
        "center" corresponds to the target instance to explain
        Segment corresponds to the size of the hypersphere
        n corresponds to the number of instances generated
        feature_variance: Array of variance for each continuous feature
    """
    #print("segment", segment)
    if segment[0] == 1 and max_features == []:
        print("IL Y A UN PROBLEME PUISQUE LE RAYON EST DE 1 et max features n'est pas initialisé", segment, center, feature_variance)
        generated_instances += 2
    d = center.shape[0]
    generated_instances = np.zeros((n,d))
    for feature, (min_feature, max_feature) in enumerate(zip(min_features, max_features)):
        range_feature = max_feature - min_feature
        # Modify the generation of artificial instance depending on the variance of each feature
        y = - segment[0] * range_feature
        z = segment[1] * range_feature
        variance = feature_variance[feature] * segment[1]
        k = (12 * variance)**0.5
        a1 = min(y, z - k)
        b1 = a1 + k
        nb_instances = int (n / 2)
        generated_instances[:nb_instances, feature] = np.random.uniform(a1, b1, nb_instances)
        generated_instances[nb_instances:, feature] = np.random.uniform(-a1, -b1, n-nb_instances)
        np.random.shuffle(generated_instances[:,feature])
    generated_instances += center
    return generated_instances

def generate_categoric_inside_ball(center, segment, percentage_distribution, n, continuous_features, categorical_features, 
                            categorical_values, min_features, max_features, feature_variance, 
                            probability_categorical_feature=None):
    """
    Generate randomly instances inside a field based on the variance of each continuous feature and 
    the maximum of distribution probability of changing a value for categorical feature
    Args: center: instance centering the field
          segment: radius of the sphere (area in which instances are generated)
          percentage_distribution: Maximum distribution probability of changing the value of categorical features
          n: Number of instances generated in the field
          continuous_features: The list of features that are discrete/continuous
          categorical_features: The list of features that categorical
          categorical_values: Array of arrays containing the values for each categorical feature
          feature_variance: Array of variance for each continuous feature 
          probability_categorical_feature: Distribution probability for each categorical features of each values 
    Return: Matrix of n generated instances perturbed randomly around center in the area of the segment based on the data distribution  
    """

    def perturb_continuous_features(continuous_features, n, feature_variance, segment, center, matrix_perturb_instances):
        """
        Perturb each continuous features of the n instances around center in the area of a sphere of radius equals to segment
        Return a matrix of n instances of d dimension perturbed based on the distribution of the dataset
        """
        d = len(continuous_features)
        generated_instances = np.zeros((n,d))
        for feature, (min_feature, max_feature) in enumerate(zip(min_features, max_features)):
            range_feature = max_feature - min_feature
            # Modify the generation of artificial instance depending on the variance of each feature
            y = - segment[0] * range_feature
            z = segment[1] * range_feature
            variance = feature_variance[feature] * segment[1]
            k = (12 * variance)**0.5
            a1 = min(y, z - k)
            b1 = a1 + k
            nb_instances = int (n / 2)
            generated_instances[:nb_instances, feature] = np.random.uniform(a1, b1, nb_instances)
            generated_instances[nb_instances:, feature] = np.random.uniform(-a1, -b1, n-nb_instances)
            np.random.shuffle(generated_instances[:,feature])
        generated_instances += center[continuous_features]
        for nb, continuous in enumerate(continuous_features):
            matrix_perturb_instances[:,continuous] = generated_instances[:,nb].ravel()
        return matrix_perturb_instances
    
    if segment[1] > 1:
        segment = list(segment)
        segment[1] = 1
        segment = tuple(segment)
    if percentage_distribution > 100:
        percentage_distribution = 100

    matrix_perturb_instances = np.zeros((n, len(center)))
    for i in range(len(categorical_features)):
        # value_libfolding generates n instances between 0 and percentage distribution 
        # (probabilities values inferior to "percentage_distribution")
        value_libfolding = np.random.uniform(0, percentage_distribution, n)
        # add for each categorical feature these values to be considered as a probability 
        matrix_perturb_instances[:, categorical_features[i]] = value_libfolding

    for nb_categorical_features, categorical_feature in enumerate(categorical_features):
        value_target_instance = center[categorical_feature]
        for nb_instance, artificial_instance in enumerate(matrix_perturb_instances):
                # if a random number is superior to the probability of the categorical feature for artificial instance
                # we do not modify its value and kept the value of the target instance
                # otherwise we generate based on the probability of distribution from the dataset one trial
                # and store the corresponding categorical value 
                if random() < artificial_instance[categorical_feature]:
                    #probability_repartition = multinomial.rvs(n=1, p=probability_categorical_feature[nb_categorical_features], size=1)[0]
                    categorical_value = np.random.choice(categorical_values[nb_categorical_features], 1, p=probability_categorical_feature[nb_categorical_features])[0]
                    
                    #categorical_value = categorical_values[nb_categorical_features][np.where(probability_repartition==1)[0][0]]
                    matrix_perturb_instances[nb_instance][categorical_feature] = categorical_value
                else:
                    matrix_perturb_instances[nb_instance][categorical_feature] = value_target_instance
    #np.set_printoptions(formatter={'float': '{:g}'.format})
    
    matrix_perturb_instances = perturb_continuous_features(continuous_features, n, feature_variance, 
                                                            segment, center, matrix_perturb_instances)
    return matrix_perturb_instances
